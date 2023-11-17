function [LowRank, Sparse] = ComputeSaliencyVideo(S,m,n,im_n)


%% Compute DMD using original method
tic
[OmegaExp,Fourierfreq,Psi,b] = Compute_Color_DMD_Full(S);

LowRankFreq = exp(min(Fourierfreq));
XDMD = zeros(m * n, im_n);
XLow = zeros(m * n, im_n);

for t = 1:im_n
    XDMD(:, t) = Psi * OmegaExp.^t * b;
end

for t = 1:im_n
    XLow(:, t) = Psi * LowRankFreq.^t * b;
end

XLow = abs(XLow);
XSparse = abs(XDMD - XLow);
XDMD = abs(XDMD);

Sparse = reshape(XSparse, [m, n, im_n]);
LowRank = reshape(XLow, [m, n, im_n]);

toc



end


