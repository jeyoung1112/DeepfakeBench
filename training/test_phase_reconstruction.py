import torch
import numpy

def phase_without_amplitude(img):
    # Convert to grayscale
    gray_img = torch.mean(img, dim=1, keepdim=True) # shape: (batch_size, 1, 256, 256)
    # Compute the DFT of the input signal
    X = torch.fft.fftn(gray_img,dim=(-1,-2))
    #X = torch.fft.fftn(img)
    # Extract the phase information from the DFT
    phase_spectrum = torch.angle(X)
    # Create a new complex spectrum with the phase information and zero magnitude
    reconstructed_X = torch.exp(1j * phase_spectrum)
    # Use the IDFT to obtain the reconstructed signal
    reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X,dim=(-1,-2)))
    # reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X))
    return reconstructed_x


img_path = ""
reconstructed_img = phase_without_amplitude(img_path)
