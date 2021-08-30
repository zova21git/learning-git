using JuMP, GLPK, LinearAlgebra

c = [-1.1;  -1; 0; 0; 0]
A = [5/3 1  1 0 0;
     8/7 1  0 1 0;
     3/7 1  0 0 1]
b = [5; 4; 3]

m, n = size(A)
x_lb = [0;0;0;0;0];

newModel = Model(with_optimizer(GLPK.Optimizer))
@variable(newModel, x[i=1:n] >=x_lb[i])
for i=1:m
    @constraint(newModel, sum(A[i,j]*x[j] for j=1:n) == b[i])
end
    @objective(newModel, Min, sum(c[j]*x[j] for j=1:n))
println("The optimization problem to be solved is:")
print(newModel)
println(" ")
println("The rank of the matrix A: ", rank(A))
println("The number of linear restrictions: ", m)
println("The number of variables: ",n)
println("Number of basic solutions n!/m!(n-m)!: ",factorial(n)/(factorial(m)*factorial(n-m)))

@time begin
    status = optimize!(newModel)
end
println("Objective value: ", JuMP.objective_value(newModel))
println("Optimal solution is x = \n", JuMP.value.(x))


using DataFrames, DataFramesMeta, Combinatorics

combs = collect(combinations(1:n, m))
resual = DataFrame(comb_1=NaN,comb_2=NaN,comb_3=NaN,x_B_1=NaN,x_B_2=NaN,x_B_3=NaN,z=NaN)
for i in 1:length(combs)
    comb = combs[i,]
    B = A[:, comb]
    c_B = c[comb]
    x_B = inv(B)*b

    if minimum(x_B)>0
        z = dot(c_B, x_B)
    else 
        z = Inf
    end
    if i==1
        resual = DataFrame(comb_1=comb[1],comb_2=comb[2],comb_3=comb[3],x_B_1=x_B[1],x_B_2=x_B[2],x_B_3=x_B[3],z=z)
    else
        push!(resual, ([comb[1],comb[2],comb[3],x_B[1],x_B[2],x_B[3],z]))
    end
end
sort(resual,cols=:z,rev=false)



