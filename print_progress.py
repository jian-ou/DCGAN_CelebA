import time

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

def run():
    totel = 309
    real = 0
    for i in range(totel):
        real = print_progress_batch(100, i, totel, real)
        time.sleep(0.1)
    print('\r                                                                                                     \r完成')

if __name__ == "__main__":
    run()