function [x11,x12,x21,x22]=pairwise(labels)

% x11,x12 denotes pairwise of must-link, x21, x22 denotes pairwise of
% cannot-link
x11=randi(720,1,5)
x12=zeros(1,5);
x21=randi(720,1,5)
x22=randi(720,1,5)

for i=1:5
   for j=1:720
       if labels(x11(i))==labels(j)
           x12(i)=j;
           break;
       end     
   end 
end
% Judge whether must-link or not
for i=1:5
    if labels(x11(i)) == labels(x12(i))
        return;
    end
end
% Judge whether cannot-link or not
for i=1:5
    if labels(x21(i))==labels(x22(i))
        return;
    end
end

end
