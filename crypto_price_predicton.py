import time
from collections import deque
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

df = pd.read_csv("datasets/LTC-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'])

currencies=["BCH-USD","BTC-USD","LTC-USD","ETH-USD"]
for cur in currencies:
    df=pd.read_csv(f"datasets/{cur}.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'])
    print(cur)
    print(df.head())
    
SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"


def label(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

full_df = pd.DataFrame()
currencies=["BCH-USD","BTC-USD","LTC-USD","ETH-USD"]
for curr in currencies:
    df=pd.read_csv(f"datasets/{curr}.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.rename(columns={"close": f"{curr}_close", "volume": f"{curr}_volume"}, inplace=True)
    df.set_index("time", inplace=True)  
    df = df.loc[:,[f"{curr}_close", f"{curr}_volume"]]

    if len(full_df)==0:  # if the dataframe is empty
        full_df = df 
    else:  
        full_df = full_df.join(df)

full_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
full_df.dropna(inplace=True)
print(full_df.head()) 

for curr in currencies:
    RATIO_TO_PREDICT=curr
    full_df['future'] = full_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
    full_df['target'] = list(map(label, full_df[f'{RATIO_TO_PREDICT}_close'], full_df['future']))
    times = sorted(full_df.index.values)  # get the times
    
    last_10pct_timestamp= sorted(full_df.index.values)[-int(0.1*len(times))]  #last 10% timestamp in times
    last_20pct_timestamp= sorted(full_df.index.values)[-int(0.2*len(times))]

    train_full_df = full_df[(full_df.index < last_20pct_timestamp)] #80% train data
    validation_full_df = full_df[(full_df.index >= last_20pct_timestamp) & (full_df.index < last_10pct_timestamp)] #10% validation
    test_full_df = full_df[(full_df.index >= last_10pct_timestamp)]# 10% test data

    def preprocess_df(crpyto_data_df):
        df = crpyto_data_df.drop("future", axis=1)  # we can drop it now as it was used to calculate target

        for col in df.columns:  # go through all of the columns
            if col != "target":  # normalize all ... except for the target itself!
                df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
                df.dropna(inplace=True)  # remove the nas created by pct_change
                df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

        df.dropna(inplace=True)  # cleanup again... jic.


        sequential_data = []  # this is a list that will CONTAIN the sequences
        prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

        for i in df.values:  # iterate over the values
            prev_days.append([n for n in i[:-1]])  # store all but the target
            if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
                sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

        random.shuffle(sequential_data)  # shuffle for good measure.

        buys = []  # list that will store our buy sequences and targets
        sells = []  # list that will store our sell sequences and targets

        for seq, target in sequential_data:  # iterate over the sequential data
            if target == 0:  # if it's a "not buy"
                sells.append([seq, target])  # append to sells list
            elif target == 1:  # otherwise if the target is a 1...
                buys.append([seq, target])  # it's a buy!

        random.shuffle(buys)  # shuffle the buys
        random.shuffle(sells)  # shuffle the sells!

        lower = min(len(buys), len(sells))  # what's the shorter length?

        buys = buys[:lower]  # make sure both lists are only up to the shortest length.
        sells = sells[:lower]  # make sure both lists are only up to the shortest length.

        sequential_data = buys+sells  # add them together
        random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

        X = []
        y = []

        for seq, target in sequential_data:  # going over our new sequential data
            X.append(seq)  # X is the sequences
            y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

        return np.array(X), y  # return X and y...and make X a numpy array!

    train_x, train_y = preprocess_df(train_full_df)
    validation_x, validation_y = preprocess_df(validation_full_df)
    test_x, test_y = preprocess_df(test_full_df)

    #set the desired EPOCHS, BATCH_SIZES and NAME of the model to be saved

    EPOCHS = 30  # how many passes through our data
    BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
    num_classes = 2 # buy and sell
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model

    num_classes = 2

    model = Sequential([
        layers.LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True),
        layers.Dropout(0.2),
        layers.BatchNormalization(), #normalizes activation outputs, same reason you want to normalize your input data.

        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.1),
        layers.BatchNormalization(),

        layers.LSTM(128),
        layers.Dropout(0.2),
        layers.BatchNormalization(),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax')
        ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    filepath = "RNN_Final-{epoch:02d}-{validation_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    #checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='validation_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    validation_x = np.asarray(validation_x)
    validation_y = np.asarray(validation_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)

    model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y),
        callbacks=[tensorboard]
        #callbacks=[tensorboard, checkpoint],
    )

    score = model.evaluate(test_x, test_y, verbose=0)
    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save model
    model.save("models/{}".format(NAME))