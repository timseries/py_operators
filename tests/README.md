%% Generate verification arrays to compare known good operator inputs and output
%%
%% Run from command line:
%%
%% $ /path/to/matlab -nosplash -nodesktop -r "run /path/to/gen_verif; quit"
%%

% Add the qbgn and toolbox files to the path, and any other matlab benchmarking code
strFilePath=[fileparts(which(mfilename('fullpath'))) '/'];
addpath([strFilePath '/3d_implicit_conv/']);
addpath([strFilePath '../../dtcwt/matlab/']);
addpath([strFilePath '../../dtcwt/matlab/qbgn/']);
addpath(genpath([strFilePath 'toolboxes/']));

%% Load Lena image and generate the quantized bandlimited gaussian noise (qbgn) phantom
inputs = load('lena.mat');
lena = inputs.lena;
qbgn = uint8(gen_qbgn(128,128));
%%load a cell image phaontom
cell_image = imreadstack([strFilePath '3d_implicit_conv/phantom_padded.tif']);

%% Generate the blur kernels in the spatial domain, 2D
blur_types={'uniform','gaussian'}; %it is important to keep this ordering in sync with python...
%blur_types=['uniform','gaussian','hamming','cylindrical','pyramid'];
blur_sizes={[9 9],[15 15], [], [7 7], []};
blur_sigmas={[],[9 9], [], [], []};
blur_kernel=cell(length(blur_types),1);
lena_blur_2D=cell(length(blur_types),1);
blur_kernel_f=cell(length(blur_types),1);
sizelena=size(lena);
H_f = zeros(size(lena));
Ny=sizelena(1);
Nx=sizelena(2);
for i = 1:length(blur_types)
    H_f = zeros(size(lena));
    blur_kernel = createBlurKernel(blur_types{i},2,blur_sizes{i},blur_sigmas{i});
    blur_kernel = blur_kernel/sum(blur_kernel(:)); %ensure unity gain
    L = (size(blur_kernel,1)-1)/2;
    H_f(Ny/2+1-L:Ny/2+1+L,Nx/2+1-L:Nx/2+1+L) = blur_kernel;
    blur_kernel_f{i} = fftn(fftshift(H_f));
end

%% Generate blur kernels in the spatial domain from files, 3D
blur_types_3D_file={[strFilePath '/3d_implicit_conv/psf.tif']};
cell_blur_3D=cell(length(blur_types_3D_file),1);
blur_kernel_3D=cell(length(blur_types_3D_file),1);
for i = 1:length(blur_types_3D_file)
    blur_kernel_3D{i}=imreadstack(blur_types_3D_file{i});
end

%%%%Generate Verfication Data%%%%%

%% Perform the ffts for verification

lena_fft = fftn(lena);
qbgn_fft = fftn(double(qbgn));

%% Perform blur in the Fourier domain and store in cell array
for i = 1:length(blur_types)
    lena_blur_2D{i} = blur_f(blur_kernel_f{i},lena_fft);
end

%% Perform blur in the Fourier domain and store in cell array, 3D files
for i = 1:length(blur_types_3D_file)
    cell_blur_3D{i} = Direct(blur_kernel_3D{i},cell_image);
end

save('verification.mat', 'lena_fft', 'qbgn_fft', 'lena_blur_2D', 'cell_blur_3D');
