local nua = require("nua")

local function finite_diff(header, eps, td, stride)
   local c = nua.mse_cost(header, td, stride)
   local nn = header["nn"]
   local Grad = nua.nn.init(header["arr_layers"])

   for k, v in pairs(nn) do
      for i = 1, #v do
         for j = 1, #v[1] do
            local temp = v[i][j]

            v[i][j] = v[i][j] + eps
            Grad[k][i][j] = (nua.mse_cost(header, td, stride) - c) / eps
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

local function train(opts)
   local header = opts.header
   local eps = opts.eps
   local rate = opts.rate
   local epoch = opts.epoch
   local step = opts.step
   local td = opts.td
   local stride = opts.stride

   for i = 1, epoch do
      local Grad = finite_diff(header, eps, td, stride)
      learn(header, Grad, rate)
      if i % step == 0 then
         print("cost = " .. nua.mse_cost(header, td, stride))
      end
   end
end

local function get_mean(ti)
   local mean = {0, 0, 0, 0}

   for i = 1, #ti do
      for j = 1, #ti[1] do
         mean[j] = mean[j] + ti[i][j]
      end
   end

   for i = 1, #mean do
      mean[i] = mean[i]/#ti
   end

   return mean
end

local function get_std(ti, mean)
   local std = {}

   for j = 1, #ti[1] do
      std[j] = 0
      for i = 1, #ti do
         local temp = ti[i][j] - mean[j]
         std[j] = std[j] + (temp*temp)
      end
   end

   for i = 1, #std do
      std[i] = std[i] / #ti
      std[i] = math.sqrt(std[i])
   end

   return std
end

local function normalise_data(ti, mean, std)
   for j = 1, #ti[1] do
      for i = 1, #ti do
         ti[i][j] = (ti[i][j] - mean[j]) / std[j]
      end
   end

   nua.mat.print(ti)
end


-- Main
local dataset = {
   {0, 0, 0, 0},
   {1, 0, 1, -1},
   {0, 1, 1, 1},
   {1, 1, 2, 0},
   {2, 1, 5, -1},
   {1, 2, 3, 3},
   {3, 1, 10, -2}
}

local mean = get_mean(dataset)
local std = get_std(dataset, get_mean(dataset))
normalise_data(dataset, mean, std)

local header = nua.nn.INIT({2, 4, 2}, {"sigmoid", 0})
local nn = header["nn"]
nua.nn.randomf(nn, 0, 5)

train({
   header = header,
   td = dataset,
   eps = 1e-2,     -- epsilon (small number)
   rate = 5e-2,    -- learning rate
   epoch = 10000,   -- amount of training
   step = 500,     -- amount of steps before printing out cost
   stride = 2
})

print("---------------------------------")
nn["a0"][1][1] = dataset[6][1]
nn["a0"][1][2] = dataset[6][2]
local output = nua.nn.forward(header)
print(string.format("1^2 + 2 = %f, 2^2 - 1 = %f", output[1][1]*std[3] + mean[3], output[1][2]*std[4] + mean[4]))


os.exit(0)
-- Print result
print("---------------------------------")
print("Xor gate: ")
for i = 0, 1 do
   for j = 0, 1 do
      nn["a0"][1][1] = i
      nn["a0"][1][2] = j
      print(string.format("%d ^ %d = %f", i, j, nua.nn.forward(header)[1][1]))
   end
end
