n = 5; % nodes per layer
len = 2; % length
nodes = 5*ones(1,len+1);  % number of nodes of each layer

x = rand(nodes(1),1); % input
y = rand(nodes(end),1); % output
v = randn(len*n^len,1); % vector

w = cell(len,1);
b = cell(len,1);

for i=1:len
    w{i}=randn(nodes(i+1),nodes(i));
    b{i}=randn(nodes(i+1),1);
end

save('hessTestRand.mat', 'x', 'y', 'w', 'b', 'v');

w1 = sym('w1_%d_%d', size(w{1}));
w2 = sym('w2_%d_%d', size(w{1}));

f = .5*sum((sigmoid(w2*sigmoid(w1*x+b{1})+b{2})-y).^2);

ghess = zeros(len*n^2,len*n^2,len*n^2);
hess = zeros(len*n^2,len*n^2);
g = zeros(len*n^2,1);

for i=1:(len*n^len)
    [x,y,z]=ind2sub([n,n,len],i);
    if z == 1
        dw1 = diff(f, w1(x,y));
    else
        dw1 = diff(f, w2(x,y));
    end
    dw = subs(dw1, [w1, w2], [w{1}, w{2}]);
    g(i) = vpa(dw);
    for j=1:(len*n^len)
        [x,y,z]=ind2sub([n,n,len],j);
        if z == 1
            dw2 = diff(dw1, w1(x,y));
        else
            dw2 = diff(dw1, w2(x,y));
        end
        dw = subs(dw2, [w1, w2], [w{1}, w{2}]);
        hess(i,j) = vpa(dw);
        for k=1:(len*n^len)
            [x,y,z]=ind2sub([n,n,len],k);
            if z == 1
                dw3 = diff(dw2, w1(x,y));
            else
                dw3 = diff(dw2, w2(x,y));
            end
            dw = subs(dw3, [w1, w2], [w{1}, w{2}]);
            ghess(i,j,k) = vpa(dw);
        end
    end
end

hv = hess*v;
vghv = squeeze(sum(ghess.*v,1));
vghv = vghv*v;

load('hessTestRand.mat');
save('hessTestRand.mat', 'x', 'y', 'w', 'b', 'v', 'ghess', 'vghv', ...
    'hess', 'hv', 'g');

function y = sigmoid(x)
    y = exp(x)./(1+exp(x));
end
