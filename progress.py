import time
import datetime

def print_progress_batch(epoch,i, epoch_size, real):
    '''
    epoch是训练次数， i是batch数
    epoch_size是尺寸
    real是上一次返回时间
    '''
    if i == 0:
        real = 0
    end_time = time.time()
    progress_num = (i / epoch_size) * 100
    real_time = end_time - real
    real_time = 1 / real_time
    train_end_time = (epoch_size - i) / real_time
    print('\r'+'Epoch  ['+str(epoch) +']  '+'完成进度：'+str(i)+' / ' + str(epoch_size) + '—>(' + str(progress_num)[0:5] + '%)  ' + str(real_time)[0:4]
                +' batch/s    大约剩余时间：'
                + str(train_end_time)[0:7] + 's            ', end='')
    return end_time

def save_time(time, dir = './data/time/time.txt'):
    '''
    传入时间和地址
    '''
    with open(dir, 'r', encoding='UTF-8') as f:
        txt = f.readlines()
        lens = len(txt)
        #print(lens)

    with open(dir, 'a+', encoding='UTF-8') as f_wirte:
        realtime = datetime.datetime.now()
        f_wirte.write("Epoch: [" + str(lens) + "]  结束运行时间：" + str(realtime) + "  耗时：" + str(time) + "s\n")

def run():
    totel = 309
    real = 0
    for i in range(totel):
        real = print_progress_batch(100, i, totel, real)
        time.sleep(0.1)
        #save_time(real)
    print('\r                                                                                                     \r完成')

if __name__ == "__main__":
    run()