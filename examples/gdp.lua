local nua = require("nua")

local function cost(header, ti, to)
   local nn = header["nn"]
   local result = 0.0

   for i = 1, #ti do
      nn["a0"][1][1] = ti[i][1]
      nn["a0"][1][2] = ti[i][2]
      nn["a0"][1][3] = ti[i][3]
      nn["a0"][1][4] = ti[i][4]

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
local ti = {
   {1.8, 4.2, 1.0, 5.0},
   {2.1, 4.0, 1.2, 5.3},
   {2.5, 3.8, 1.5, 5.5},
   {3.0, 4.1, 1.8, 5.2},
   {3.5, 4.5, 2.0, 4.8},
   {2.8, 4.3, 1.7, 4.3},
   {2.2, 4.0, 1.4, 4.6},
   {1.9, 3.9, 1.2, 5.0},
   {1.5, 3.7, 1.0, 5.4},
   {1.2, 3.5, 0.8, 5.8},
   {1.0, 3.4, 0.6, 6.2},
   {1.3, 3.6, 0.7, 6.5},
   {1.7, 3.8, 0.9, 6.3},
}

local to = {
   {5.3},
   {5.5},
   {5.2},
   {4.8},
   {4.3},
   {4.6},
   {5.0},
   {5.4},
   {5.8},
   {6.2},
   {6.5},
   {6.3},
   {6.0}
}

local mean = get_mean(ti)
get_std(ti, get_mean(ti))
normalise_data(ti, mean, get_std(ti, mean))

local gdp = nua.nn.INIT({4, 6, 1}, {"relu", 0})
local nn = gdp["nn"]
nua.nn.randomf(nn, 0, 7)

local eps = 1e-2     -- epsilon (small number)
local rate = 8e-5    -- learning rate
local epoch = 100000   -- amount of training
local step = 10000     -- amount of steps before printing out cost
train(gdp, eps, rate, epoch, step, ti, to)

-- Results

nn["a0"][1][1] = ti[10][1]
nn["a0"][1][2] = ti[10][2]
nn["a0"][1][3] = ti[10][3]
nn["a0"][1][4] = ti[10][4]
print(string.format("%f, %f, %f, %f = %f", ti[10][1], ti[10][2], ti[10][3], ti[10][4], nua.nn.forward(gdp)))

nn["a0"][1][1] = ti[4][1]
nn["a0"][1][2] = ti[4][2]
nn["a0"][1][3] = ti[4][3]
nn["a0"][1][4] = ti[4][4]
print(string.format("%f, %f, %f, %f = %f", ti[4][1], ti[4][2], ti[4][3], ti[4][4], nua.nn.forward(gdp)))
