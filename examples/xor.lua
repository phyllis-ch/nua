local nua = require("nua")

-- Modify as needed
local function cost(header, ti, to)
   local nn = header["nn"]
   local result = 0.0

   for i = 1, #ti do
      nn["a0"][1][1] = ti[i][1]
      nn["a0"][1][2] = ti[i][2]

      local y = nua.nn.forward(header)
      local d = y - to[i][1]
      result = result + d*d
   end
   return result / #ti
end

local function finite_diff(header, eps, ti, to)
   local c = cost(header, ti, to)
   local nn = header["nn"]
   local Grad = nua.nn.init(header["arr_layers"])

   for k, v in pairs(nn) do
      for i = 1, #v do
         for j = 1, #v[1] do
            local temp = v[i][j]

            v[i][j] = v[i][j] + eps
            Grad[k][i][j] = (cost(header, ti, to) - c) / eps
            v[i][j] = temp
         end
      end
   end

   return Grad
end

-- Apply diffs
local function learn(header, Grad, rate)
   local nn = header["nn"]

   for k, v in pairs(nn) do
      for i = 1, #v do
         for j = 1, #v[1] do
            v[i][j] = v[i][j] - (rate * Grad[k][i][j])
         end
      end
   end
end

local function train(header, eps, rate, epoch, step, ti, to)
   for i = 1, epoch do
      local Grad = finite_diff(header, eps, ti, to)
      learn(header, Grad, rate)
      if i % step == 0 then
         print("cost = " .. cost(header, ti, to))
      end
   end
end



-- Main

local training_input = {
   {0, 0},
   {0, 1},
   {1, 0},
   {1, 1}
}

local training_output = {
   {0},
   {1},
   {1},
   {0}
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
local header = nua.header_create({2, 2, 1}, {"sigmoid", "relu"})
-- To make it less crowded, a variable that points to the neural network can be created
local Xor = header["nn"]
nua.nn.randomf(Xor, 0, 1)

-- TODO: A function to interact with the array will probably be better
-- header["arr_func"] = { "sigmoid", "relu" }

local eps = 1e-3     -- epsilon (small number)
local rate = 1e-1    -- learning rate
local epoch = 5000   -- amount of training
local step = 500     -- amount of steps before printing out cost
train(header, eps, rate, epoch, step, training_input, training_output)

-- Print result
print("---------------------------------")
print("Xor gate: ")
for i = 0, 1 do
   for j = 0, 1 do
      Xor["a0"][1][1] = i
      Xor["a0"][1][2] = j
      print(string.format("%d ^ %d = %f", i, j, nua.nn.forward(header)))
   end
end
