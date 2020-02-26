import numpy as np

def sigmoid(x):
	return 1/ (1 + np.exp(-x))

#Training Data-------------------------------------------
training_inputs = np.array([[0,0,0,1,1,1,0,0,0],
							[0,1,1,1,1,1,1,1,0],
							[0,1,0,0,0,0,0,1,0],
							[1,1,0,0,0,0,0,1,1],
							[1,1,0,0,0,0,0,1,1],
							[1,1,0,0,0,0,0,1,1],
							[0,1,0,0,0,0,0,1,0],
							[0,1,1,1,1,1,1,1,0],
							[0,0,0,1,1,1,0,0,0]])

training_outputs = np.array([[0,0,0,0,0,0,0,0,0],
							 [0,0,0,0,0,0,0,0,0],
						     [0,0,1,1,1,1,1,0,0],
						 	 [0,0,1,1,1,1,1,0,0],
							 [0,0,1,1,1,1,1,0,0],
							 [0,0,1,1,1,1,1,0,0],
							 [0,0,1,1,1,1,1,0,0],
							 [0,0,0,0,0,0,0,0,0],
							 [0,0,0,0,0,0,0,0,0]])

if training_inputs.any():
	if training_outputs.any():
		print("Training Data loaded!")

print()
#--------------------------------------------------------


#Initializing random weights-----------------------------
np.random.seed(1)

synaptic_weights = 2 * np.random.random((9,9)) - 1

if synaptic_weights.any():
	print("Random weights initialized!")
print()
#--------------------------------------------------------


#Back Propagation Method-------------------
print("Learning starts...")
for i in range(1000000):
	input_layer = training_inputs
	outputs = sigmoid( np.dot(input_layer, synaptic_weights) )

	err = training_outputs - outputs
	adjustments = np.dot( input_layer.T, err*(outputs*(1-outputs)) ) #ZDES' VSYA SUT' KORREKTIROVKI VESOV, err*outputs eto KOEFFICIENT dlya polucheniya popravki (adjustments)
	
	synaptic_weights += adjustments

print("Learning Complete!")
#--------------------------------------------------------

print( "Result on train Data:" )
outputs_clear = np.round_(outputs)
print( outputs_clear )

#Task----------------------------------------------------
#---Task Data
print()
print("Loading Task Data...")
input_layer = np.array([[0,0,0,0,1,1,0,0,0],
						[0,0,0,1,1,1,1,0,0],
						[0,1,1,1,0,0,1,1,0],
						[1,1,0,0,0,0,0,1,1],
						[1,1,0,0,0,0,0,1,1],
						[1,1,0,0,0,0,0,1,1],
						[0,1,0,0,0,0,1,1,0],
						[0,0,1,1,1,0,1,0,0],
						[0,0,0,1,1,1,0,0,0]])

#---Applying
print("Applying Task Data to model...")
outputs = sigmoid( np.dot(input_layer, synaptic_weights) )
outputs_clear = np.round_(outputs)
#----------

#---Output
print("Done! Result for gain Task:" )
if np.array_equal(outputs_clear,training_outputs):
	print("Shape is: ○")
else:
	print("Shape is unknown!")
#---------
#--------------------------------------------------------