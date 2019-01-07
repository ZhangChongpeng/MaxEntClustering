function A = constructEntropy(X, k)
H=X;
alpha=100;
k=13;
[~,n]=size(H);
for i=1:n
    for j=1:n
        e(i,j)=norm(H(:,i)-H(:,j),2)^2;
    end
end

mol=exp(-(e/(2*alpha)));
for i=1:n
    di=sort(mol(i,:),'descend');
    for j=1:n
        if mol(i,j)<di(k+1)&&i~=j
            mol(i,j)=0;
        end
        if i==j
            mol(i,j)=0;
        end
    end
end
A=mol./(sum(mol,2));
A = (A+A')/2;

[a,b]=size(A);
for i=1:a
    for j=1:b
        if A(i,j)~=0
            A(i,j)=1;
        end
    end
end