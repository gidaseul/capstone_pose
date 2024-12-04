import queue
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from time import time_ns
from typing import TypeVar, Generic, Optional, final

T = TypeVar("T")
I = TypeVar("I")
O = TypeVar("O")


class Pipe(Generic[T]):
    def __init__(self):
        self.queue: Queue[tuple[T, list[tuple[int, int]]]] = Queue(maxsize=1)

    def get(self) -> tuple[T, list[tuple[int, int]]]:
        return self.queue.get()

    def put(self, data: T, time_list: list[tuple[int, int]]):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put((data, time_list))


class PipedProcess(ABC, Process, Generic[I, O]):
    def __init__(self):
        super().__init__()
        self.input_pipe: Optional[Pipe[I]] = None
        self.output_pipe: Optional[Pipe[O]] = None
        self.daemon = True

    @final
    def run(self):
        self.init()
        while True:
            input_data, time_list = self.input_pipe.get() if self.input_pipe else (None, [])
            start_time = time_ns()
            output_data = self.process(input_data)
            end_time = time_ns()
            time_list.append((start_time, end_time))
            if self.output_pipe:
                self.output_pipe.put(output_data, time_list)
            else:
                for i in range(len(time_list)):
                    s, e = map(lambda x: x / 10 ** 6, time_list[i])

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def process(self, input_data: I) -> O:
        pass


PP = TypeVar("PP", bound=PipedProcess)


class Pipeline:
    def __init__(self, *processes: PipedProcess):
        self.pipe: tuple[Pipe, ...] = tuple(Pipe() for _ in range(len(processes) - 1))
        self.processes: tuple[PipedProcess, ...] = processes

        for i in range(len(processes) - 1):
            processes[i].output_pipe = processes[i + 1].input_pipe = self.pipe[i]

    def start(self):
        for process in self.processes:
            process.start()
        try:
            while True:
                # 마지막 프로세스에서 종료 요청 확인
                if hasattr(self.processes[-1], 'stop_pipeline') and self.processes[-1].stop_pipeline:
                    #print("Pipeline stopping...")
                    break
        # except KeyboardInterrupt:
        #     print("Pipeline interrupted")
        finally:
            self.stop_all_processes()

    def stop_all_processes(self):
        for process in self.processes:
            process.kill()



class PipelineWithLogging(Pipeline):
    def start(self):
        for process in self.processes:
            process.start()
