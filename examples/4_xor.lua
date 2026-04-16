local nua = require("nua")

local training_data = {
   {0, 0, 0},
   {0, 1, 1},
   {1, 0, 1},
   {1, 1, 0}
}

local header = nua.nn.new({2, 2, 1}, {"sigmoid", "relu"})
local Xor = header["nn"]
nua.nn.randomf(Xor, 0, 1)

-- TODO: A function to interact with the array will probably be better
-- header["arr_func"] = { "sigmoid", "relu" }

nua.train({
   header = header,
   eps = 1e-3,
   rate = 1e-1,
   epoch = 5000,
   step = 500,
   td = training_data,
   stride = 2
})

-- Print result
print("---------------------------------")
print("Xor gate: ")
for i = 0, 1 do
   for j = 0, 1 do
      Xor["a0"][1][1] = i
      Xor["a0"][1][2] = j
      print(string.format("%d ^ %d = %f", i, j, nua.nn.forward(header)[1][1]))
   end
end
