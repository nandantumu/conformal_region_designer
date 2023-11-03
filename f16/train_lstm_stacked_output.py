


import pickle

import numpy as np
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

from myLSTM import LSTMNew

useTrainedModel = False






def sliding_windows_stack_pred(data,prediction_future_window,skip_steps):

    x = []
    y = []

        
    for i in range(len(data)):
        _x = []
        _y = []
        for j in range(0,len(data[i])-prediction_future_window*skip_steps,skip_steps):
            _x.append(data[i][j])
            __y = []
            for k in range(prediction_future_window):
                for data_type_ind in range(len(data[i][j+skip_steps+k*skip_steps])): ## pattern is [type_1_time_0,type_2_time_0,type_1_time_1,type_2_time_1,...]
                    __y.append(data[i][j+skip_steps+k*skip_steps][data_type_ind]) ## SEEME: should the [0] be [:] instead?
            _y.append(__y)
        x.append(_x)
        y.append(_y)

    
    return np.array(x),np.array(y)


save_scaler = True
save_train_test = True

def main():


    ## load the data
    with open('data/all_data_3400_tmax_5.pkl','rb') as f:
        all_data = pickle.load(f)

    ## turn into multi dim arrays with the inputs and outputs
    # for now, just train on altitude and vertical velocity (as both inputs and outputs)
    # input_types = ['alt','vt','theta']
    input_types = ['alt','theta']
    output_types = ['alt','theta']

    print("Num traces in dataset: " + str(len(all_data[input_types[0]])))



    train_size = 1500 #int(len(all_data[input_types[0]]) * 0.67)
    # train_size = int(len(all_data[input_types[0]]) * 0.99)


    all_data_one_array = [] ## only used for fitting the sklearn scaler
    for i in range(train_size):
        for j in range(len(all_data[input_types[0]][i])):
            all_data_one_array.append([all_data[input_type][i][j] for input_type in input_types])




    train_data = np.array([[all_data[input_types[j]][i] for j in range(len(input_types))] for i in range(train_size)])

    print(train_data.shape)

    sc = MinMaxScaler()
    _ = sc.fit_transform(all_data_one_array)

    print(len(all_data_one_array))
    print(len(all_data_one_array[0]))
    print(all_data_one_array[0])
    # input("wait")
    
    if save_scaler:
        with open("models/firstTryLSTMScaler_alt_theta.pkl","wb") as f:
            pickle.dump(sc,f)


    trace_data = [[sc.transform([[all_data[input_type][i][j] for input_type in input_types]])[0] for j in range(len(all_data[input_types[0]][i]))] for i in range(len(all_data[input_types[0]]))]
    # trace_data = [[[[all_data[input_type][i][j] for input_type in input_types]][0] for j in range(len(all_data[input_types[0]][i]))] for i in range(len(all_data[input_types[0]]))]

    prediction_future_window = 25#20#10#2#5#10
    skip_steps = 3#5
    x,y = sliding_windows_stack_pred(trace_data,prediction_future_window,skip_steps) ## TODO: theres a bug in here

    # print(x[0][0:6])
    # print(y[0][0:2])

    print(x.shape)
    print(y.shape)

    # input("wait")

    # train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    print("Number of training data: " + str(train_size))
    print("Number of testing data: " + str(test_size))

    print(trainX.shape)
    print(trainY.shape)

    ## save train test data
    if save_train_test:
        x_train = x[0:train_size]
        y_train = y[0:train_size]

        x_test = x[train_size:len(x)]
        y_test = y[train_size:len(y)]

        data_folder = 'data/stackedOutput/3400DataPoints/pred_future_window_' + str(prediction_future_window) + '/skip_step_' + str(skip_steps)
        os.makedirs(data_folder,exist_ok = True)
        with open(data_folder + '/x_train.pkl','wb') as f:
            pickle.dump(x_train,f)
        with open(data_folder + '/y_train.pkl','wb') as f:
            pickle.dump(y_train,f)
        with open(data_folder + '/x_test.pkl','wb') as f:
            pickle.dump(x_test,f)
        with open(data_folder + '/y_test.pkl','wb') as f:
            pickle.dump(y_test,f)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    num_epochs = 1000#500
    learning_rate = 0.01

    input_size = len(input_types)
    lstm_hidden_size = 25
    hidden_size = 25#10#2
    lstm_num_layers = 2#1

    print(trainY.shape)
    num_classes = trainY.shape[-1]#len(input_types) ## TODO: this is wrong!!!
    print("Num classes: " + str(num_classes))

    # lstm = LSTM(num_classes, input_size, hidden_size, num_layers,device=device)
    lstm = LSTMNew(num_classes, input_size,lstm_hidden_size, hidden_size, lstm_num_layers,device=device)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    model_folder = 'models/stackedOutput/3400DataPoints/pred_future_window_' + str(prediction_future_window) + '/skip_step_' + str(skip_steps)
    os.makedirs(model_folder,exist_ok = True)
    model_path = model_folder + '/firstTryLSTM_alt_theta_hAndc.pt'
    # model_path = 'models/firstTryLSTM_alt_hAndc.pt'
    lstm.to(device)
    trainX = trainX.to(device)
    trainY = trainY.to(device)

    # print(trainX)
    if not useTrainedModel:
        # Train the model
        for epoch in range(num_epochs):
            outputs,_,_ = lstm(trainX)
            optimizer.zero_grad()

            # print(outputs.size())
            
            # obtain the loss function
            loss = criterion(outputs, trainY)
            
            loss.backward()
            
            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


        torch.save(lstm.state_dict(), model_path)
    else:
        lstm.load_state_dict(torch.load(model_path))
        lstm.to(device)

    lstm.eval()

    ## compute test error
    testX = testX.to(device)
    testY = testY.to(device)

    test_outputs,_,_ = lstm(testX)
    loss = criterion(test_outputs, testY)
    print("Test loss: %1.5f" % (loss.item()))



    trials_to_plot = testX.shape[0]
    all_errs_one_point = []

    for j in range(trials_to_plot):

        sample_data = testX[j:j+1,:,:]
        sample_output = testY[j:j+1,:,:]

        sample_predict,_,_ = lstm(sample_data)
        sample_predict = sample_predict.view(sample_data.size(1),-1)

        _sample_predict_plot = sample_predict.cpu().data.numpy()
        _sample_output_plot = sample_output.view(sample_data.size(1),-1).cpu().data.numpy()

        new_preds = []
        new_gts = []
        for i in range(_sample_predict_plot.shape[0]):
            curr_pred = _sample_predict_plot[i,:]
            temp_preds = []
            for k in range(0,len(curr_pred),2):
                temp_preds.append([curr_pred[k],curr_pred[k+1]]) ## SEEME: this is now hardcoded for 2 outputs
            new_preds.append(temp_preds)
        
            curr_gt = _sample_output_plot[i,:]
            temp_gts = []
            for k in range(0,len(curr_gt),2):
                temp_gts.append([curr_gt[k],curr_gt[k+1]]) ## SEEME: this is now hardcoded for 2 outputs
            new_gts.append(temp_gts)

        pred_of_interest = new_preds[0][-1]
        gt_of_interest = new_gts[0][-1]

        resid = [pred_of_interest[0]-gt_of_interest[0],pred_of_interest[1]-gt_of_interest[1]]
        all_errs_one_point.append(resid)

        # pred_of_interest_gt = 
        
    image_dir = 'images/stackedOutput/3400DataPoints/pred_future_window_' + str(prediction_future_window) + '/skip_step_' + str(skip_steps) + '/alt_theta/'
    os.makedirs(image_dir,exist_ok=True)
    plt.scatter([r[0] for r in all_errs_one_point],[r[1] for r in all_errs_one_point])
    plt.xlabel("Altitude error")
    plt.ylabel("Pitch error")
    plt.savefig(image_dir + "scatterPlotTestErrs")
    plt.clf()
            
        
        # # print(_sample_predict_plot.shape)
        # # input("wait")
        # sample_predict_plot = sc.inverse_transform(_sample_predict_plot)
        # sample_output_plot = sc.inverse_transform(_sample_output_plot)

        # print(sample_output_plot.shape)
        # input("wait")


        # labels_types = ['altitude','pitch','velocity']

        # image_dir = 'images/stackedOutput/3400DataPoints/pred_future_window_' + str(prediction_future_window) + '/skip_step_' + str(skip_steps) + '/samplePredictions_trial_' + str(j)
        # os.makedirs(image_dir,exist_ok=True)
        # for i,input_type in enumerate(input_types):
            
        #     for j in range(len(sample_predict_plot)):
        #         # plt.plot(list(range(j,j+prediction_future_window)),[s[j:j+prediction_future_window] for s in sample_predict_plot],'b-',label='predict')
        #         plt.plot([s[i] for s in sample_output_plot],'r-',label='GT Trace')
        #         plt.plot(list(range(j,j+prediction_future_window)),sample_output_plot[j],'g-',label='GT Output')
        #         plt.plot(list(range(j,j+prediction_future_window)),sample_predict_plot[j],'b-',label='predict')
                
        #         plt.legend()

        #         plt.title('Time-Series Prediction: ' + labels_types[i])
        #         plt.savefig(image_dir +  '/samplePred_alt_' + input_type + '_' + str(j) + '.png')

        #         plt.clf()

if __name__ == '__main__':
    main()
