function [OmegaExp,Fourierfreq,Psi,b] = Compute_Color_DMD_Full(D)

X1 = D(:,1:(end-1)); 
X2 = D(:,2:end);

[U, E, V] = svd(X1, 'econ');
r = rank(E);

% Truncated versions of U, E, and V
U = U(:,1:r);
E = E(1:r,1:r);
V = V(:,1:r);

Stild = U' * X2 * V * pinv(E);
[EigVec, EigVal] = eig(Stild);

%DMD Spectra
Psi = ((X2*V)/E)*EigVec; %DMD Modes %Psi = U * EigVec;
OmegaExp = exp(log(EigVal)); %complex
Fourierfreq = abs(log(diag(EigVal)));
b = pinv(Psi)*X1(:,1);

end

