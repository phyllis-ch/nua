local nua = require("nua")

local dataset = {
   {0, 0},
   {1, 2},
   {2, 4},
   {3, 6},
   {4, 8}
}

-- nua.header_create takes in a table of integers, and an optional array of activation functions
-- Table count will be the layers of neural network 
-- Each element of the table are amount of neurons in each layer
-- First element will always be amount of input
-- Last element will always be amount of output
--
-- nua.header_create creates a header
-- A header consists of
-- 1. array of layers passed as input
-- 2. array of activation functions
-- 3. a pointer to the actual neural network

-- Amount of activation functions should always be one less than length of array of layers
-- As in #func = #layers-1

-- In this case, this is a 1 neuron to 1 neuron network
local nn_h = nua.nn.new({1, 1}, {0})
-- To make it less crowded, a variable that points to the neural network can be created
local nn = nn_h["nn"]
nua.nn.randomf(nn, 0, 10)

nua.train({
   header = nn_h,
   td = dataset,
   eps = 1e-3,    -- epsilon
   rate = 1e-3,   -- learning rate
   epoch = 8000,  -- amount of training
   step = 500,    -- amount of steps before printing out cost
   stride = 1
})

-- Print result
print("---------------------------------")
for i = 0, 10 do
   nn["a0"][1][1] = i
   local output = nua.nn.forward(nn_h)
   print(string.format("%d * 2 = %f", nn["a0"][1][1], output[1][1]))
end
