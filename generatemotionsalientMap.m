function sl_map=generatemotionsalientMap(ft_map)

sl_param.size=[0.1,0.5];
ft_F=dct2(ft_map);

%S=fftclshift(log(1+abs(ft_F)));
%figure,imshow(S,[])

[n,m]=size(ft_F); %filter size

fq=1./sl_param.size;  
ef=max(fq);  %delta
sf=min(fq);                
kernelF=gaussFilterFq([n,m],[0,0],[ef,ef])-gaussFilterFq([n,m],[0,0],[sf,sf]);


sl_map=idct2(ft_F.*kernelF);
sl_map=sl_map./max(sl_map(:));


% gaussian1 = fspecial('Gaussian', 21, 2);
% gaussian2 = fspecial('Gaussian', 21, 10);
% dog = gaussian1 - gaussian2;
% sl_map = conv2(double(ft_map), dog, 'same');
% sl_map=sl_map./max(sl_map(:));

end
