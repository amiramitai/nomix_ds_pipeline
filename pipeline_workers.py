import multiprocessing
import os
import queue
import sys
import time
import threading
import traceback

from myqueue import EMPTY_QUEUE_SLEEP, QUEUE_MAX_SIZE

KEEP_ALIVE_WORKER_SLEEP = 0.2

PROGRESSBAR_STAGES = 10
WHITE_BG = '\033[5;30;47m'
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


def print_summary(start, stages):
    final_str = []
    time_passed = int(time.time() - start)
    total_occup_threads = 0
    for s in stages:
        input_info = ('{:%dd}'%(PROGRESSBAR_STAGES)).format(s.input_queue.qsize())
        fract = s.input_queue.qsize() / QUEUE_MAX_SIZE
        level = int(fract * PROGRESSBAR_STAGES)
        
        formatted = [WHITE_BG]
        formatted.append(input_info[:level])
        formatted.append(NC)
        formatted.append(input_info[level:])
        input_info = ''.join(formatted)
        
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
    final_str.append('[{}]'.format(s.output_queue.counter.value))
    final_str.append(' t[{}/{}] '.format(total_occup_threads, multiprocessing.cpu_count() ))
    final_str.append(str(time_passed))
    print('\r' + ''.join(final_str), end='')


def is_main_thread_alive():
        return any([a.is_alive() for a in threading.enumerate() if a.name == 'MainThread'])


def prioritize_threads(stages):
    cpu_count = multiprocessing.cpu_count()
    used_threads = sum([s.get_occupied_threads() for s in stages])
    alloced_threads = sum([s.get_thread_alloc() for s in stages])
    avail_threads = cpu_count - (used_threads + alloced_threads)
    if avail_threads == 0:
        return
    if used_threads >= cpu_count:
        return
    
    stages_to_prior = []
    priorities = []
    for s in stages:
        current_alloc = s.get_occupied_threads() + s.get_thread_alloc()
        if s.input_queue.qsize() == 0:
            continue

        if s.get_max_parallel() == 1:
            if current_alloc >= 1:
                continue
            s.add_thread_alloc(1)
            avail_threads -= 1
            continue
        
        input_priority = s.input_queue.qsize() / QUEUE_MAX_SIZE
        output_priority = 1.0 - (s.output_queue.qsize() / QUEUE_MAX_SIZE)
        current_alloc = s.get_occupied_threads() + s.get_thread_alloc()
        priority = input_priority * output_priority
        priorities.append(priority)
        stages_to_prior.append(s)

    if not stages_to_prior:
        return
    
    m = max(priorities)
    if m > 0:
        npriorities = []
        for p in priorities:
            npriorities.append((1.0/m) * p)
        priorities = npriorities
    
    zipped = zip(stages_to_prior, priorities)
    zipped = sorted(zipped, key=lambda k: k[1])
    zipped.reverse()
    for i in range(10):
        if avail_threads == 0:
            return
        for s, priority in zipped:
            current_alloc = s.get_occupied_threads() + s.get_thread_alloc()
            if current_alloc > s.get_max_parallel():
                continue
            t_to_alloc = int(priority * avail_threads)
            if t_to_alloc > s.get_max_parallel():
                t_to_alloc = max(s.get_max_parallel() - current_alloc, 0)
            
            t_to_alloc = min(t_to_alloc, avail_threads)
            s.add_thread_alloc(t_to_alloc)
            avail_threads -= t_to_alloc


@keyboard_int_guard
def keepalive_worker(keepalive, stages):
    start = time.time()
    while is_main_thread_alive():
        keepalive.value = time.time()
        prioritize_threads(stages)
        print_summary(start, stages)
        time.sleep(KEEP_ALIVE_WORKER_SLEEP)


@keyboard_int_guard
def write_thread_occup_guard(start_event, thread_sem, stage, item):
    thread_sem.acquire()
    try:
        stage.use_thread_slot()
        if start_event:
            start_event.set()
        stage.write(item)
    finally:
        stage.free_thread_slot()
        thread_sem.release()


STAGE_GET_TIMEOUT = 5000
KEEP_ALIVE_TIMEOUT = 10000
@keyboard_int_guard
def no_fork_pipeline_stage_worker(thread_sem, stage, keepalive):
    stage.in_proc_init()
    while True:
        try:
            if stage.input_queue.qsize() == 0 or stage.get_thread_alloc() == 0:
                time.sleep(EMPTY_QUEUE_SLEEP)
                continue
            
            item = stage.input_queue.get(True, STAGE_GET_TIMEOUT)
            write_thread_occup_guard(None, thread_sem, stage, item)
        except queue.Empty:
            if (time.time() - keepalive.value) < KEEP_ALIVE_TIMEOUT:
                print('[+] keep alive timeout({}).. killing..'.format(stage.name))
                return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()


@keyboard_int_guard
def pipeline_stage_worker(thread_sem, stage, keepalive):
    stage.in_proc_init()
    spawned_procs = []
    while True:
        try:
            alive = []
            for p in spawned_procs:
                if p.is_alive():
                    alive.append(p)
                else:
                    del p
            spawned_procs = alive
            thread_alloc = stage.get_thread_alloc()
            if stage.input_queue.qsize() == 0 or thread_alloc == 0:
                time.sleep(EMPTY_QUEUE_SLEEP)
                continue
            for t in range(thread_alloc):
                item = stage.input_queue.get(True, STAGE_GET_TIMEOUT)
                proc_event = multiprocessing.Event()
                p = multiprocessing.Process(target=write_thread_occup_guard, args=(proc_event, thread_sem, stage, item))
                p.start()
                proc_event.wait(STAGE_GET_TIMEOUT)
                spawned_procs.append(p)
        except queue.Empty:
            if (time.time() - keepalive.value) < KEEP_ALIVE_TIMEOUT:
                print('[+] keep alive timeout({}).. killing..'.format(stage.name))
                return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()