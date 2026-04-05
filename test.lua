local mat = require("matrix")

-- Test fill
local td = mat.init(2, 2)
local td2 = mat.init(2, 2)
mat.fill(td, 5)
mat.fill(td2, 5)

-- Test print
print("td: ")
mat.print(td)
print("td2: ")
mat.print(td2)

mat.sum(td, td2)
print("sum: ")
mat.print(td)

-- Test random
local test = mat.init(3, 3)
mat.randomf(test, 5, 10)
mat.print(test)

-- Test multiplication
local tb = {
   {2, 2, 2, 2},
   {2, 2, 2, 2},
   {2, 2, 2, 2},
}
local tb2 = {
   {2, 2, 2},
   {2, 2, 2},
   {2, 2, 2},
   {2, 2, 2},
}
local dst = mat.init(3, 3)
mat.dot(dst, tb, tb2)
print("dot: ")
mat.print(dst)
