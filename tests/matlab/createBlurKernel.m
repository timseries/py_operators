function H = createBlurKernel(strType,intD,arySize,arySigma)
% CREATEBLURKERNEL - Create a blur kernel with the correct dimension, size, and type.
%   
    switch strType
      case 'uniform'
        switch intD
          case 1
            H=ones(arySize(1),1)*1/(arySize(1));  
          case 2
            H=ones(arySize(1),arySize(2))*1/prod(arySize);              
          case 3
            H=ones(arySize(1),arySize(2),arySize(3))*1/prod(arySize);              
        end
      case 'hamming'
        switch intD
          case 1
            H=hamming(arySize(1));            
          case {2,3}
            error('unsupported dimension for hamming window');
        end
      case 'gaussian'
        switch intD
          case 1
            dblWidth=max(arySize(1),round((6*arySigma(1)-1)/2));            
            arySupport = (-dblWidth:dblWidth);
            aryGaussian = exp(-(arySupport.^2)./(2*arySigma(1)^2));
          case 2
            if arySigma(1)==arySigma(2)
                H=fspecial('gaussian',arySize,arySigma(1));
            else
                error('no code support for 2d asymmetric gaussian');
            end
          case 3
            H = nonIsotropicGaussianPSF(arySigma,max(max(arySize(:)),max(2.1*arySigma(:))));
        end
%taken from J. Portilla's icip 2009 package
      case 'cylindrical'  
        switch inD
          case 1
            error('no code support for 2d asymmetric gaussian');
          case 2 
            if arySigma(1)==arySigma(2)
                i = meshgrid(-arySize(1):-arySize(1), -arySize(2):-arySize(2)); j = i'; 
                H = 1./(i.^2 + j.^2 + 1);    % PSF 1         case 2
            else
                error('no code support for 2d asymmetric gaussian');
            end
          case 3
            error('no code support for 2d asymmetric gaussian');
        end 
      case 'pyramid'  
        switch inD
          case 1
            error('no code support for 2d asymmetric gaussian');
          case 2 
            if arySigma(1)==arySigma(2)
                H = [1 4 6 4 1]'*[1 4 6 4 1]/256; % PSF 3
            else
                error('no code support for 2d asymmetric gaussian');
            end
          case 3
            error('no code support for 2d asymmetric gaussian');
        end
    end      
end 
    

