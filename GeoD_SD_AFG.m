clc;clear;

d = 1e4;
x0 = ones(d, 1);
alpha = 0.5;
beta = 1;
learning_rate = 0.1;
etol = 1e-8;
maxiter = 1000;

error_GeoD = GeoD(d, alpha, x0, maxiter, etol);
error_SD = SD(d, learning_rate, x0, maxiter, etol);
error_AFG = AFG(d, learning_rate, x0, maxiter, etol);

% compare
s1 = semilogy(error_GeoD);
hold on
s2 = semilogy(error_SD);
s3 = semilogy(error_AFG);
ylim([1e-8 1e4]);
xlabel("Number of iterations");
ylabel("Error");
title("GeoD & SD & AFG: β = 1, d = 1e4");
legend([s1, s2, s3], "GeoD: α = 0.5", "SD: η = 0.1", "AFG: η = 0.1");

% seperate
% subplot(1, 3, 1);
% semilogy(error_GeoD);
% ylim([1e-8 1e4]);
% xlabel("Number of iterations");
% ylabel("Error");
% title("GeoD");
% 
% subplot(1, 3, 2);
% semilogy(error_SD);
% ylim([1e-8 1e4]);
% xlabel("Number of iterations");
% ylabel("Error");
% title("SD");
% 
% subplot(1, 3, 3);
% semilogy(error_AFG);
% ylim([1e-8 1e4]);
% xlabel("Number of iterations");
% ylabel("Error");
% title("AFG");


function error = GeoD(d, alpha, x0, maxiter, etol)
error = zeros(maxiter + 1, 1);
xopt = optimal(d);
x0_ = line_search(x0, x0 - df(x0));
c0 = x0 - df(x0)/alpha;
R0 = norm(df(x0))^2/alpha^2 - 2/alpha*(f(x0) - f(x0_));

xl = x0;
xl_ = x0_;
cl = c0;
Rl = R0;
iter = 0;
fprintf("Start GeoD, the error is %.8f!\n", norm(xopt - x0)^2);
error(1) = norm(xopt - x0)^2;
while norm(xopt - xl)^2 >= etol
    iter = iter + 1;
    if iter > maxiter
        fprintf("Fail to converge!\n");
        return
    end

    % combining step
    x = line_search(xl_, cl);

    % gradient step
    x_ = line_search(x, x - df(x));

    % ellipsoid step
    xA = x - df(x)/alpha;
    RA = norm(df(x))^2/alpha^2 - 2/alpha*(f(x) - f(x_));
    xB = cl;
    RB = Rl - 2/alpha*(f(xl_ - f(x_)));
    [c, R] = min_enclose_ball(xA, RA, xB, RB);

    % update
    xl = x;
    xl_ = x_;
    cl = c;
    Rl = R;
    error(iter + 1) = norm(xopt - xl)^2;
    % output
    fprintf("After %d iteration, the point error is %.8f.\n", iter, error(iter + 1));
end
fprintf("Achieve point 1e-8-close after %d iteration!\n\n", iter);
error = error(1:iter + 1);
end


function error = SD(d, learning_rate, x0, maxiter, etol)
error = zeros(maxiter + 1, 1);
xopt = optimal(d);
iter = 0;
fprintf("Start SD, the error is %.8f!\n", norm(xopt - x0)^2);
error(1) = norm(xopt - x0)^2;
x = x0;
while norm(xopt - x)^2 >= etol
    iter = iter + 1;
    if iter > maxiter
        fprintf("Fail to converge!\n");
        break
    end

    g = df(x);
    x = x - learning_rate*g;

    error(iter + 1) = norm(xopt - x)^2;
    % output
    fprintf("After %d iteration, the point error is %.8f.\n", iter, error(iter + 1));
end
fprintf("Achieve point 1e-8-close after %d iteration!\n\n", iter);
error = error(1:iter + 1);
end


function error = AFG(d, learning_rate, x0, maxiter, etol)
error = zeros(maxiter + 1, 1);
xopt = optimal(d);
iter = 0;
fprintf("Start AFG, the error is %.8f!\n", norm(xopt - x0)^2);
error(1) = norm(xopt - x0)^2;
x = x0;
a0 = 0.1;
a_pre = a0;
while norm(xopt - x)^2 >= etol
    iter = iter + 1;
    if iter > maxiter
        fprintf("Fail to converge!\n");
        break
    end

    g = df(x);
    x_ = x - learning_rate*g;
    q = 0.5;
    a = bisection(@(a)a.^2-(1-a)*(a_pre.^2)-q*a,0,1);
    
    b_k = a_pre*(1-a_pre)/(a_pre.^2+a);
    x = x_ +b_k*(x_ - x);
    error(iter + 1) = norm(xopt - x)^2;
    a_pre = a;
    % output
    fprintf("After %d iteration, the point error is %.8f.\n", iter, error(iter + 1));
end
fprintf("Achieve point 1e-8-close after %d iteration!\n\n", iter);
error = error(1:iter + 1);
end


function xopt = optimal(d)
A = zeros(d, d);
A(1, 1) = 3;
A(1, 2) = -1;
for i = 2:d
    A(i, i - 1) = 1;
    A(i, i) = -3;
    if i ~= d
        A(i, i + 1) = 1;
    end
end
b = zeros(d, 1);
b(1) = 1;
xopt = A\b;
end


function l = line_search(l, r)
tol = 1e-8;
while norm(l - r)^2 >= tol 
    mid = (l + r)/2;
    if f(mid) < f(l)
        l = mid;
    elseif f(mid) < f(r)
        r = mid;
    end
end
end


function [c, R] = min_enclose_ball(xA, RA, xB, RB)
if norm(xA - xB)^2 >= abs(RA - RB)
    c = (xA + xB)/2 - (RA - RB)/2/norm(xA -xB)^2*(xA -xB);
    R = RB - (norm(xA - xB)^2 + RB - RA)^2/4/norm(xA - xB)^2;
elseif norm(xA - xB)^2 < RA - RB
    c = xB;
    R = RB;
else
    c = xA;
    R = RA;
end
end


function y = f(x)
beta = 1;
d = length(x);
y = beta/2*((1 - x(1))^2 + sum((x(1:d - 1) - x(2:d)).^2) + x(d)^2) + sum(x.^2)/2;
end


function g = df(x)
beta = 1;
d = length(x);
g = zeros(d, 1);
g(1) = beta*(x(1) - 1 + x(1) - x(2)) + x(1);
g(2:d - 1) = beta*(2*x(2:d - 1) - x(1:d - 2) - x(3:d)) + x(2:d - 1);
g(d) = beta*(x(d) - x(d - 1) + x(d)) + x(d);
end


function [x, fx, exitFlag] = bisection(f, lb, ub, target, options)
tolX = 1e-6;
tolFun = 0;
if nargin == 5
    if isstruct(options)
        if isfield(options,'TolX') && ~isempty(options.TolX)
            tolX = options.TolX;
        end 
        if isfield(options,'TolFun') && ~isempty(options.TolFun)
            tolFun = options.TolFun;
        end
    else
        tolX = options;
    end      
end

if (nargin < 4) || isempty(target)
    target = 0;
else
    f = @(x) f(x) - target;
end

%% Flip UB and LB elements if necessary. 
isFlipped = lb > ub;
if any(isFlipped,'all')
    [ub(isFlipped),lb(isFlipped)] = deal(lb(isFlipped),ub(isFlipped));
end

%% Make sure everything is the same size for a non-scalar problem. 
fub = f(ub);
if isscalar(lb) && isscalar(ub)
    % Test if f returns multiple outputs for scalar input.
    if ~isscalar(target)
        id = ones(size(target));
        ub = ub.*id;
        fub = fub.*id;
    elseif ~isscalar(fub)
        ub = ub.*ones(size(fub));
    end
end

% Check if lb and/or ub need to be made into arrays.
if isscalar(lb) && ~isscalar(ub)    
    lb = lb.*ones(size(ub));
elseif ~isscalar(lb) && isscalar(ub)
    id = ones(size(lb));
    ub = ub.*id;
    fub = fub.*id;
end

unboundedRoot = sign(fub).*sign(f(lb)) > 0;
ub(unboundedRoot) = NaN;

%% Iterate
ubSign = sign(fub);  
while true
    x = (lb + ub) / 2;
    fx = f(x);
    outsideTolX = abs(ub - x) > tolX;
    outsideTolFun = abs(fx) > tolFun;
    stillNotDone = outsideTolX & outsideTolFun;
    if ~any(stillNotDone(:))
        break;
    end
    select = sign(fx) ~= ubSign;
    lb(select) = x(select);
    ub(~select) = x(~select);
end

%% Catch NaN elements of UB, LB, target, or other funky stuff. 
x(isnan(fx)) = NaN;
x(isnan(tolX)) = NaN;
x(isnan(tolFun)) = NaN;
fx(isnan(x)) = NaN;

%% Characterize results. 
if nargout > 1 && nnz(target(:))
    fx = fx + target;
end
if nargout > 2 
    exitFlag                                    = +~outsideTolX;
    exitFlag(~outsideTolFun)                    =  2;
    exitFlag(~outsideTolFun & ~outsideTolX)     =  3;
    exitFlag(isnan(x))                          = -1;
    exitFlag(unboundedRoot)                     = -2;
end
end
