data = read.csv("realtor-data.zip.csv")

data = na.omit(data)

sqr_footage = data$bed
cost = data$price

m = mean(cost)

print(m)

plot(sqr_footage, cost, xlim=c(0,5), ylim=c(0,1.0e+07))
help("plot")

r = cor(sqr_footage, cost)
print(r)

x = c(data$bed)
y = c(data$price)
xbar = mean(x)
ybar = mean(y)
sxx = sum( (x -xbar)^2) 
sxy = sum ( (x - xbar)*(y - ybar) )
syy = sum ( (y - ybar)^2 )

b = sxy / sxx
a = ybar - b*xbar

print(b)
print(a)

lin_regr <- function(x){
  a + b*x
} 

print(lin_regr(3))


