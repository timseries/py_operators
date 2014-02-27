function y = blur_f(x_f, kernel_f)
    y = real(ifftn(x_f.*kernel_f));
end
