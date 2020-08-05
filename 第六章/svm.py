from svmutil import *

train_name = 'train.txt'
test_name = 'test.txt'
linear_name = 'linear.model'
gauss_name = 'gauss.model'

y,x = svm_read_problem(train_name)
y1,x1 = svm_read_problem(test_name)

model_1 = svm_train(y,x,'-t 0')  ### linear
model_2 = svm_train(y,x,'-t 2')   ### radial basis function

svm_save_model(linear_name,model_1)
svm_save_model(gauss_name,model_2)

p1_label, p1_acc, p1_val = svm_predict(y1,x1,model_1)
p2_label, p2_acc, p2_val = svm_predict(y1,x1,model_2)
