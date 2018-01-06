import multiprocessing
import os
import queue
import sys
import time
import threading
import traceback
import math
import xmlrpc.client
import pickle

from myqueue import EMPTY_QUEUE_SLEEP
from exceptions import NoThreadSlots

KEEP_ALIVE_WORKER_SLEEP = 0.2

GREEN = '\033[1;32;40m'
YELLOW = '\033[1;33;40m'
NC = '\033[0m'

def keyboard_int_guard(func):
    def _inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print('[!] Recieved KeyboardInterrupt.. exit(1)', os.getpid())
            sys.exit(1)

    return _inner

def remote_pipe_send_worker(config, context):
    print('[+] remote_pipe_send_worker thread started..')
    servers = {}
    for remote in config['remotes']:
        uri = 'http://{}:{}'.format(remote['host'], remote['port'])
        name = remote['name']
        print('[+] connecting to {}/{}'.format(name, uri))
        servers[name] = xmlrpc.client.ServerProxy(uri)
    while is_main_thread_alive():
        try:
            for name, pipe in context.remotes.items():
                while pipe.qsize() > 0:
                    remote_stage_name, item = pipe.get()
                    server = servers[name]
                    server.put_remote(remote_stage_name, pickle.dumps(item))
            time.sleep(5.0)
        except:
            traceback.print_exc()
            time.sleep(2.0)

#shutil.get_terminal_size((80, 20))
def print_summary(start, stages):
    final_str = []
    time_passed = int(time.time() - start)
    total_occup_threads = 0
    for s in stages:
        input_info = s.input_queue.get_input_info()
        
        input_pipe = '[{}]>>'.format(input_info)
        if s.index == 0:
            input_pipe = ''
        if not s.should_display:
            final_str.append(input_pipe)
            continue
        occup_threads = s.get_occupied_threads()
        total_occup_threads += occup_threads
        alloc_threads = s.get_thread_alloc()
        state_color = NC
        if occup_threads > 0:
            state_color = GREEN
        elif alloc_threads > 0:
            state_color = YELLOW
        pipe_rep = '{}{}[{}|{}]{}'.format(state_color, s.name, alloc_threads, occup_threads, NC)
        final_str.append('{}{}>>'.format(input_pipe, pipe_rep))
    final_str.append('[{}]'.format(s.output_queue.get_input_info()))
    # final_str.append('[{}]'.format(s.output_queue.counter.value))
    final_str.append(' t[{}/{}] '.format(total_occup_threads, multiprocessing.cpu_count() ))
    final_str.append(str(time_passed))
    # print('\r' + ''.join(final_str), end='')
    print(''.join(final_str))


def is_main_thread_alive():
        return any([a.is_alive() for a in threading.enumerate() if a.name == 'MainThread'])


def prioritize_threads(stages):
    cpu_count = multiprocessing.cpu_count()
    priorities = []
    max_priority = 0.001
    # get the priority for each stage is work available 
    for s in stages:
        input_priority = s.input_queue.qsize() / s.input_queue.get_capacity()
        output_priority = 1.0 - (s.output_queue.qsize() / s.output_queue.get_capacity())
        priority = input_priority * output_priority

        if s.get_max_parallel() > 1:
            max_priority = max(max_priority, priority)
            priorities.append(priority)
        else:
            priorities.append(priority)            

    stages_to_prior = list(reversed(sorted(zip(stages, priorities), key=lambda k: k[1])))
    allocs = []
    avail_threads = cpu_count
    # each stage which has work, allocate at least one thread
    for s, priority in stages_to_prior:
        if avail_threads == 0:
            break

        # if priority > 0:
        #     allocs.append(1)
        #     avail_threads -= 1
        # else:
        allocs.append(0)
    
    # for the rest of avail threads, allocate by priority
    for i, (s, priority) in enumerate(stages_to_prior):
        cur_avail_threads = min(avail_threads, s.get_max_parallel() - allocs[i])
        t_to_alloc = int((1.0 / max_priority) * priority * cur_avail_threads)

        allocs[i] += t_to_alloc
        avail_threads -= t_to_alloc
        # print('[+] {}/{} - {}: {} threads [MT]'.format(i, len(stages_to_prior), s.name, allocs[i]))
        s.set_thread_alloc(allocs[i])    
        
    return


# @keyboard_int_guard
def keepalive_worker(context, stages):
    start = time.time()
    while is_main_thread_alive():
        context.keepalive.value = time.time()
        prioritize_threads(stages)
        print_summary(start, stages)
        time.sleep(KEEP_ALIVE_WORKER_SLEEP)


# @keyboard_int_guard
def write_thread_occup_guard(start_event, thread_sem, stage, cancel, profiler):        
    free_slot = True
    try:
        thread_sem.acquire()
        stage.use_thread_slot()


        if start_event:
            start_event.set()
        while True:
            
            if profiler:
                with profiler.record('{}.input_queue.get'.format(stage.name)):
                    item = stage.input_queue.get(True, STAGE_GET_TIMEOUT)
                with profiler.record('{}.write'.format(stage.name)):
                    stage.write(item)
            else:
                item = stage.input_queue.get(True, STAGE_GET_TIMEOUT)
                stage.write(item)
            
            if cancel and not cancel.is_set():
                continue
            
            if cancel and cancel.is_set():
                # print('[BYE!] {} got cancel [MT]'.format(stage.name))
                break

            break
    except queue.Full:
        pass
        # print('[!] {}: queue is full!'.format(stage.name))
    except NoThreadSlots:
        pass
        # print('[!] {}: no thread slots!'.format(stage.name))
        free_slot = False
    finally:   
        # print('[BYE!] {} is done [MT]'.format(stage.name))
        if cancel:
            cancel.set()
        
        if start_event:
            start_event.set()

        if free_slot:
            stage.free_thread_slot()
        thread_sem.release()
        # print('[+] {}: before finally release'.format(stage.name))
        # thread_sem.release()
        # print('[+] {}: after finally release'.format(stage.name))


STAGE_GET_TIMEOUT = 5000
KEEP_ALIVE_TIMEOUT = 10000
# @keyboard_int_guard
def no_fork_pipeline_stage_worker(init_barrier, thread_sem, stage, context):
    stage.in_proc_init()
    init_barrier.wait()
    while True:
        try:
            if stage.input_queue.qsize() == 0 or stage.get_thread_alloc() == 0:
                time.sleep(EMPTY_QUEUE_SLEEP)
                continue
            write_thread_occup_guard(None, thread_sem, stage, None, context.profiler)
            stage.output_queue.refresh_cache()
        except queue.Empty:
            if (time.time() - context.keepalive.value) < KEEP_ALIVE_TIMEOUT:
                print('[+] keep alive timeout({}).. killing..'.format(stage.name))
                return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()


# @keyboard_int_guard
def pipeline_stage_worker(init_barrier, thread_sem, stage, context):
    stage.in_proc_init()
    init_barrier.wait()
    worker_cancel_events = []
    while True:
        try:
            thread_alloc = stage.get_thread_alloc()
            thread_occup = stage.get_occupied_threads()

            worker_cancel_events = [e for e in worker_cancel_events if not e.is_set()]
            if thread_occup > thread_alloc:
                # threads need to finish
                if thread_alloc < len(worker_cancel_events):
                    events_to_cancel = len(worker_cancel_events) - thread_alloc
                    # print('[XXXX] canceling in {}: {} events [MT]'.format(stage.name, events_to_cancel))
                    for e in worker_cancel_events[:events_to_cancel]:
                        e.set()
                    time.sleep(EMPTY_QUEUE_SLEEP)
                    continue

                # print('[+] {}: waiting for alloc > occupied [MT]'.format(stage.name))
                time.sleep(EMPTY_QUEUE_SLEEP)
                continue
            if thread_alloc == thread_occup:
                # print('[+] {}: no need for new allocation.. keep working.. [MT]'.format(stage.name))
                time.sleep(EMPTY_QUEUE_SLEEP)
                continue
            if stage.input_queue.qsize() == 0 or thread_alloc == 0:
                # print('[+] {}: empty queue.. skipping.. [MT]'.format(stage.name))
                time.sleep(EMPTY_QUEUE_SLEEP)
                continue

            to_spawn = thread_alloc - len(worker_cancel_events)
            # print('[++++] {}: allocating {} threads [MT]'.format(stage.name, to_spawn))
            for t in range(to_spawn):
                proc_event = multiprocessing.Event()
                cancel_event = multiprocessing.Event()
                worker_cancel_events.append(cancel_event)
                kwargs = {
                    'start_event': proc_event,
                    'thread_sem': thread_sem,
                    'stage': stage,
                    'cancel': cancel_event,
                    'profiler': context.profiler,
                }
                multiprocessing.Process(target=write_thread_occup_guard, kwargs=kwargs).start()
                proc_event.wait(STAGE_GET_TIMEOUT)
            stage.output_queue.refresh_cache()
        except queue.Empty:
            if (time.time() - context.keepalive.value) < KEEP_ALIVE_TIMEOUT:
                print('[+] keep alive timeout({}).. killing..'.format(stage.name))
                return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()