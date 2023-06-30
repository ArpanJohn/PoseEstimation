import threading
import os
import keyboard


# #create a simple threadding class
class MyThread():

    def __init__(self):
        super(MyThread, self).__init__()
        self.myList = []
        self.some_number = 0

    def append_list(self):
        while True:
            if keyboard.is_pressed('q'):
                break
            self.myList.append(5)
            self.some_number += 1

    def print_list(self):
        while True:
            if keyboard.is_pressed('q'):
                break
            print(len(self.myList), self.some_number)

    def run(self):
        t1 = threading.Thread(target=self.append_list)
        t2 = threading.Thread(target=self.print_list)
        t1.start()
        t2.start()
        t1.join()
        t2.join()



if __name__ == '__main__':

    thread = MyThread()
    thread.run()  