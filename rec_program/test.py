from multiprocessing import Process, Queue, freeze_support

class A:
    def __init__(self, vl):
        self.vl = vl

    def cal(self, nb, q):
        result = nb * self.vl
        q.put(result)

    def run(self):
        q = Queue()
        processes = []
        p1 = Process(target=self.cal, args=(1, q))
        p2 = Process(target=self.cal, args=(1, q))
        p1.start()
        p2.start()
        processes.append(p1)
        processes.append(p2)

        for p in processes:
            p.join()

        results = []
        while not q.empty():
            results.append(q.get())

        return results

if __name__ == '__main__':
    freeze_support()

    a = A(2)
    print(a.run())
