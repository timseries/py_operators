%% Generate verification arrays to compare known good DT-CWT inputs and outputs.
%%
%% This script assumes that Nick Kingsbury's DT-CWT toolbox has been installed.
%%
%% Run with something like:
%%
%% $ /path/to/matlab -nosplash -nodesktop -r "run /path/to/gen_verif; quit"
%%
%% There should be the DTCWT toolboxes installed in a toolboxes directory next
%% to the script. See the regen_verification.sh script for an example of setting
%% this up.

% Add the qbgn and toolbox files to the path
strFilePath=[fileparts(which(mfilename('fullpath'))) '/dtcwt/matlab/'];
addpath([strFilePath 'qbgn/']);

%% Load Lena image and generate the quantized bandlimited gaussian noise (qbgn) phantom
inputs = load('lena.mat');
lena = inputs.lena;
qbgn = uint8(gen_qbgn(128,128));

%% Generate the blur kernels in the spatial domain, 2D
blur_types={'uniform','gaussian'};
%blur_types=['uniform','gaussian','hamming','cylindrical','pyramid'];
blur_sizes={[9 9],[15 15], [], [7 7], []};
blur_sigmas={[],[9 9], [], [], []};
blur_kernel=cell(length(blur_types),1);
blur_kernel_f=cell(length(blur_types),1);
sizelena=size(lena);
H_f = zeros(size(lena));
Ny=sizelena(1);
Nx=sizelena(2);
for i in 1:length(blur_types)
    blur_kernel = createBlurKernel(blur_types(i),2,blur_sizes(i),blur_sigmas(i));
    blur_kernel = blur_kernel/sum(blur_kernel(:));
    L = (size(blur_kernel,1)-1)/2;
    H_f(Ny/2+1-L:Ny/2+1+L,Nx/2+1-L:Nx/2+1+L) = blur_kernel;
    blur_kernel_f(i) = fftn(fftshift(H_f));
end
%% Perform the ffts for verification

lena_fft = fftn(lena);
qbgn_fft = fftn(double(qbgn));

%% Perform blur in the Fourier domain
for i in 1:length(blur_types)
    switch blur_types(i)
      case 'uniform'
        lena_blur_uniform = blur_f(blur_kernel_f(i),lena_fft);
      case 'gaussian'
        lena_blur_gaussian = blur_f(blur_kernel_f(i),lena_fft);
    end    
end    

save('verification.mat', 'lena_fft', 'qbgn_fft', 'lena_blur_uniform', 'lena_blur_gaussian');
