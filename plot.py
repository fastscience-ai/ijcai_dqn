import numpy as np
import matplotlib.pyplot as plt


def plot(arr1, arr2):
    cnt = 1
    step = []
    perf1=[]
    perf2=[]
    for p1, p2 in zip(arr1, arr2):
            step.append(cnt)
            perf1.append(p1)
            perf2.append(p2)
            cnt += 1
    plt.xlabel("testing episodes")
    plt.ylabel("perfomance")
    plt.plot(step, perf1, 'b-', label = '8.h5f(0.84)')
    plt.plot(step, perf2, 'r-', label = 'fine-tuned in 200(0.933)')
    plt.legend()
    plt.savefig("testing_curve_200.png")
    plt.close()

def plot_test(arr1):
    cnt = 1
    step = []
    perf1=[]
    for p1 in arr1:
            print(p1)
            step.append(cnt)
            perf1.append(p1)
            cnt += 1
    plt.xlabel("testing episodes")
    plt.ylabel("perfomance")
    plt.plot(step, perf1, 'b-', label = 'model_200 -> testing 101(0.20192)')
    plt.axvline(x = 10, color = 'r', label = 'novelty applied')
    plt.legend()
    plt.savefig("testing_curve_101.png")
    plt.close()

def plot_train(arr1):
    cnt = 1
    step = []
    perf1=[]
    for p1 in arr1:
            print(p1)
            step.append(cnt)
            perf1.append(p1)
            cnt += 1
    #plt.xlim([0, 510])
    plt.xlabel("training episodes")
    plt.ylabel("perfomance")
    plt.plot(step, perf1, 'b-', label = 'model_200 -> training 101')
    plt.axvline(x = 10, color = 'r', label = 'novelty applied')
    plt.legend()
    plt.savefig("training_curve_101.png")
    plt.close()


def plot_train_and_test(arr1):
    cnt = 11
    step = [i+1 for i in range(10)]
    perf1=[1.0 for i in range(10)]
    for p1 in arr1:
            #print(p1)
            step.append(cnt)
            perf1.append(p1)
            cnt += 1
    #plt.xlim([0, 510])
    plt.xlabel("traing episodes [x10] ")
    plt.ylabel("averaged testing perfomance over 10 episodes")
    plt.plot(step, perf1, 'bs-', label = 'model_200 -> retraining 101 and testing 101 (train 10 ep/one timestep)')
    plt.axvline(x = 10, color = 'r', label = 'novelty applied')
    plt.legend()
    plt.savefig("training_10episodes_testing_10points_curve_101.png")
    plt.close()


def text_to_array(file):
    perf=[1.0 for i in range(10)] # novelty applied at timestep=10
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            perf.append(float(line.split(",")[0])/200.0)
        perf=np.asarray(perf)
    np.save("retrain_101.npy", perf)



arr1=np.load("test_reward_200_0.8408199999999999.npy")
arr2 =np.load("test_reward_200_0.9330499999999999.npy") 
arr=np.load("agent_A.npy")
plot(arr1, arr2)
plot_test(arr)
text_to_array("training_reward.txt")
arr_retrain=np.load("retrain_101.npy")
plot_train(arr_retrain)
arr_soo=np.load("test_reward_101_while_training_for_100ts.npy")
plot_train_and_test(arr_soo)

