"""
train.py

training DeepDesigner
Writer: EJYang (yejyang@kaist.ac.kr)
Last Update: 2017/12/30
"""
from evaluator import *
import tensorflow as tf

batchSize = 32
trainSize = 525
layer.BatchSize = batchSize
learning_rate = 0.01
EVAL_MODEL_PATH = "../model/evaluator"
LABEL_PATH = "./deep_designer_objective.csv"
IS_TRAIN = True


# Open CSV File and prepare labelList
labelList = []
readyTrainingData(LABEL_PATH, labelList)
IMG1, IMG2, labelset, shape = getTrainingData(batchSize=batchSize, labelList=labelList,trainSize=len(labelList))

# evaluator model
I1, I2, Y, flatten, fc3, cost = evaluator(batchSize, shape, is_train=IS_TRAIN)
optimizer = (tf.train.AdagradOptimizer(learning_rate=learning_rate, use_locking=False, name='optimizer')).minimize(cost)

# ready to train
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

# restore evaluator model
if IS_TRAIN == True:
    sess.run(init)
    saver.restore(sess, EVAL_MODEL_PATH)
    step = 57400
else:
    saver.restore(sess, EVAL_MODEL_PATH)
    print sess


# Training
if IS_TRAIN == True:
    train_loss = 1.0e+10
    outfile = open("evaluator_train_out.txt", "w")
    tIMG1, tIMG2, tbatch_y, _ = getTrainingData(labelList=labelList,batchSize=batchSize,
                                                trainSize=len(labelList), IsValidate=True)

    while train_loss >= 1e-9:
        IMG1, IMG2, batch_y, _ = getTrainingData(labelList=labelList, batchSize=batchSize,
                                                 trainSize=len(labelList))
        sess.run(optimizer, feed_dict={I1: IMG1, I2: IMG2, Y: batch_y})

        if step % 10 == 0:
            train_loss, convout, scores = sess.run((cost, fc3, flatten),
                                                   feed_dict={I1: IMG1, I2: IMG2, Y: batch_y})
            print ("====step : %d, train loss %f " % (step, train_loss))
            outfile.write("====step : %d, train loss %f \n" % (step, train_loss))

        if step % 100 == 0:
            saver.save(sess, EVAL_MODEL_PATH)

        step += 1

# Validation
else:

    printIdx = 0
    tIMG1, tIMG2, tbatch_y, _ = getTrainingData(labelList=labelList,batchSize=batchSize,
                                                trainSize=len(labelList), IsValidate=True)
