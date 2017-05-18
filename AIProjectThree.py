import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot

test_x = []
test_y = []

def main():
	print("Running AI Three")
	# Get the training and testing data from the input files
	training_set_1 = get_training_data(1)
	training_set_2 = get_training_data(2)
	training_set_3 = get_training_data(3)
	testing_set_1  = get_testing_data()
	
	# Learning constant
	learn   = 0.035
	
	# Random weights
	weights = [ random.random(), random.random() ]
	
	# Consider normalizing inputs before training and testing
	combined_array = training_set_1 + training_set_2 + training_set_3
	combined_norm_array, temp_min, temp_max = normalize_data(combined_array, 0, 0)
	norm_testing_set, mi, ma = normalize_data(testing_set_1, temp_min, temp_max)
	
	# Create and train perceptron
	p = Perceptron(learn, weights)
	p.train_neuron(combined_norm_array, "norm")
	p.test_neuron(norm_testing_set, "norm")

	plot_data(combined_array)
	plot_normalized_data(combined_norm_array)
	plot_testing_data(norm_testing_set)
	

# Plot the error
def plot_error(iterations, errors):
    print("Plotting Error over iterations")
    figure = plt.figure(1)
    figure.suptitle("AI Project Three Error VS Time", fontsize=16, fontweight="bold")
    plt.title("Wes Bosman")
    plt.xlabel("Iteration")
    plt.ylabel("Total Error")
    plt.plot(iterations, errors, linewidth=2, color="orange")
    plt.savefig("AIProjectThree_Error_VS_Time")
    plt.show()


# Plot the Non Normalized Input
def plot_data(input_data):
	print("Creating Graph")
	figure = plt.figure(2)
	figure.suptitle("AI Project Three", fontsize=16, fontweight="bold")
	plt.title("Wes Bosman")
	plt.xlabel("Time of Day")
	plt.ylabel("Temperature")
	plot_data_from_input(input_data)
	plt.legend(loc="upper left")
	plt.savefig("AIProjectThree.png")
	plt.show()


# Plot the Normalized Input
def plot_normalized_data(input_data):
	print("Creating Normalized Graph")
	figure = plt.figure(3)
	figure.suptitle("AI Project Three Normalized", fontsize=16, fontweight="bold")
	plt.title("Wes Bosman")
	plt.xlabel("Time Of Day")
	plt.ylabel("Temperature")
	plot_data_from_input(input_data)
	plt.legend(loc="upper left")
	plt.savefig("AIProjectThreeNormalized.png")
	plt.show()


# Plot the Testing Data and Predictions
def plot_testing_data(input_data):
	print("Creating Testing Graph")
	figure = plt.figure(4)
	figure.suptitle("AI Project Three Normalized Testing Data", fontsize=16, fontweight="bold")
	plt.title("Wes Bosman")
	plt.xlabel("Time Of Day")
	plt.ylabel("Temperature")
	test_time = []
	test_temp = []
	
	for i in range(0, len(input_data)):
		time = input_data[i][0]
		temp = input_data[i][2]
		test_time.append(time)
		test_temp.append(temp)
	
	plt.scatter(test_time, test_temp, color="purple", marker="o", label="Day 4 Test Data")
	plt.scatter(test_x , test_y, color="green" , marker="x", label="Day 4 Predictions")
	plt.legend(loc="upper left")
	plt.savefig("AIProjectThreeNormalizedTesting.png")
	plt.show()



# Plot the Testing Data
def plot_data_from_input(input_data):
	train_1_time = []
	train_1_temp = []
	train_2_time = []
	train_2_temp = []
	train_3_time = []
	train_3_temp = []

	for i in range(0, len(input_data)):
		time = input_data[i][0]
		temp = input_data[i][2]
		if i >= 0 and i <= 8:
			train_1_time.append(time)
			train_1_temp.append(temp)
		elif i >= 9 and i <= 17:
			train_2_time.append(time)
			train_2_temp.append(temp)
		else:
			train_3_time.append(time)
			train_3_temp.append(temp)

	plt.scatter(train_1_time, train_1_temp, marker="^", c="red",   label="Day 1")
	plt.scatter(train_2_time, train_2_temp, marker="o", c="green", label="Day 2")
	plt.scatter(train_3_time, train_3_temp, marker="s", c="blue",  label="Day 3")
		

# Normalize the data
def normalize_data(inputs, t_min, t_max):
	# Arrays to hold temperatures and times
	norm_temps = []
	temps = []
	hours = []
	
	# Loop through the data
	for i in range(0, len(inputs)):
		hour = inputs[i][0]
		bias = inputs[i][1]
		temp = inputs[i][2]
		temps.append(temp)
		hours.append(hour)
	
	# Set Maximum temp
	if t_max == 0:
		temp_max = max(temps)
	else:
		temp_max = t_max
		
	# Set Minimum temp
	if t_min == 0:
		temp_min = min(temps)
	else:
		temp_min = t_min
		
		
	diff = temp_max - temp_min
	
	for (tme, tmp) in zip(hours, temps):
		time = tme
		temp = tmp
		norm_x = (temp - temp_min) / diff
		norm_temps.append([time, 1, norm_x])

	# Return the normalized temperatures
	return norm_temps, temp_min, temp_max
		

# Get The Training Data
def get_training_data(i):
	train_data = []

	file_name = "train_data_%d.txt" %(i)
	with open(file_name) as file:
		print("Reading Data from %s" %(file_name))
		for line in file:
			current_line = line.split(",")
			time = int(current_line[0])
			temp = float(current_line[1])
			bias = 1
			train_data.append([time, bias, temp])

	return train_data
	

# Get The Testing Data
def get_testing_data():
	test_data = []
	file_name = "test_data_4.txt"
	with open(file_name) as file:
		print("Reading Data from %s" %(file_name))
		for line in file:
			current_line = line.split(",")
			time = int(current_line[0])
			temp = float(current_line[1])
			bias = 1
			test_data.append([time, bias, temp])
	return test_data
			
	
# Create a perceptron
class Perceptron(object):
	def __init__(self, learning_constant, weights):
		self.learning_constant = learning_constant
		self.weights = weights
		self.max_iterations = 10000
		self.training_complete = False
		self.testing_error = 0.0
		
		
	# Train neuron
	def train_neuron(self, x, type_of_pattern):		
		iterations = 0
		output = np.zeros(len(x))

		# Keep track of iterations and errors
		num_iterations  = []
		iteration_error = []
		
		# While iterations is not max or minimum error not reached
		while(self.training_complete == False):
			# Count the number of iterations
			iterations += 1
			total_error = 0.0
            print("")
            print("Training Iteration %d " %iterations)
            print("")
            print("|------------------------------------------------------------------------|")
            print("| Time |  Learn  |  Net  | Predict Temp | Actual Temp | Error | % Change | ")
            print("|------------------------------------------------------------------------|")
			
			# Number of items in list 
			for i in range(0, len(x)):
				net = 0
				
				# For each item in the list 
				# Get the sum of the weights times the input including the bias
				for j in range(0, len(x[i]) - 1):
					net = net + self.weights[j] * x[i][j]
				
				# Get values from array
				time = x[i][0]
				bias = x[i][1]
				temp = x[i][2]
				
				# Calculate the net of the activation function
				if type_of_pattern == "norm":
					output[i] = self.norm_activation_function(net)
				else:
					output[i] = self.activation_function(net)
				
				# Calculate error
				error = temp - output[i]
				learn = self.learning_constant * error
				
				# This is percent change between the desired and predicted values
				percent_change = ((output[i] - temp) / temp) * 100 

				# Calculate error get the total error 
				total_error = total_error + abs(error)


				self.print_training_table(time, learn, net, output[i], temp , error, percent_change)

				# Update the weights 
				for k in range(0, len(x[1]) - 1):
					self.weights[k] = self.weights[k] + ( learn * x[i][k] )


			print("Total Error = %.6f" % total_error)

			# Add the number of iterations and the error to arrays to plot them
			num_iterations.append(iterations)
			iteration_error.append(total_error)
			
			# If the max_iterations have been reached break the loop or error < 0.00001
			if(iterations == self.max_iterations or total_error <= 0.00001):
				self.training_complete = True
				print("Stopped the while loop a condition was met")
				print("Iterations    = %d   " %iterations)
                plot_error(num_iterations, iteration_error)


	
	
	# Test the neuron on the 4th dataset
	def test_neuron(self, x, type_of_pattern):
		print(" ")
		print("Testing Neuron")
		print("|----------------------------------------------------------------------------------|")
		print("| Time 	  |    Predicted Temp 	|     Actual Temp     |     Error     |  % Change  |")
		print("|----------------------------------------------------------------------------------|")
		for j in range(0, len(x)):
			net = 0
			
			for k in range(0, len(x[j]) - 1):
				net = net + self.weights[k] * x[j][k]
			
			time = x[j][0]
			bias = x[j][1]
			temp = x[j][2]
			# append the time to the global array
			test_x.append(time)
		
			if type_of_pattern == "norm":
				prediction = self.norm_activation_function(net)
				# append the prediction to the global array
				test_y.append(prediction)
			else:
				prediction = self.activation_function(net)
			
			# Get the absolute value of the error
			error 		   = abs(temp - prediction)
			self.testing_error = self.testing_error + error
			
			self.print_testing_table(time, prediction, temp)
		print("Testing Error                 = %.6f" %(self.testing_error))
		print("Testing Error Average Percent = %.6f%%" %((self.testing_error / len(x)) * 100))

				

	# Print the testing data in a table 
	def print_testing_table(self, time, prediction, temp):
		# If the time is after 11 AM its the Afternoon print PM
		if(time > 11):
			print_time = " %d:00%s  |" %(time, "PM")
		else:
			if(time > 9):
				print_time = " %d:00%s  |" %(time, "AM")
			else:
				print_time = " %d:00%s   |" %(time, "AM")

		error 	      = temp - prediction
		print_predict = "       %.4f        |" %prediction
		print_temp    = "       %.4f        |" %temp
		print_error   = "    %.4f     |" %(error)
		print_err_per = "    %.2f     |" %(((prediction - temp)/temp) * 100)
		output        = print_time + print_predict + print_temp + print_error + print_err_per
		print(output)



	# Print Training Table 
	def print_training_table(self, time, learn, net, predict, actual, error, err_percent):
		print_time    = str(" %d:00  " %time)
		print_lrn     = str(" %.4f  " %learn).ljust(6, " ")
		print_net     = str(" %.2f  " %net).rjust(8, " ") 
		print_pred    = str(" %.2f  " %predict).rjust(13, " ")
		print_act     = str(" %.2f  " %actual).rjust(12, " ")
		print_err     = str(" %.4f  " %error).rjust(12, " ")
		print_err_per = str("   %.4f   " %err_percent).rjust(6, " ")
		output = print_time + print_lrn + print_net + print_pred + print_act + print_err + print_err_per
		print(output)



	# Linear Activation Function
	def activation_function(self, net):
		y = net * 5.88 + 43
		return y
		
	# Should this function be parabolic?
	def norm_activation_function(self, net):
		y = net * 0.3 - 0.4
		return y
		
		
# Run the Program
main()


if __name__ == main:
	main()
