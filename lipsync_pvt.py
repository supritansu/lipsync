!nvcc --version

from google.colab import drive
drive.mount('/content/gdrive')

!pip install pytube

import os
from pytube import YouTube

# Function to download a video from a URL with the highest resolution
def download_highest_resolution_video(url, output_path):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if stream:
            stream.download(output_path=output_path)
            print(f"Downloaded: {yt.title} (Resolution: {stream.resolution})")
        else:
            print(f"No MP4 video available for {yt.title}")

    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

# Function to read video URLs from a file and download them with the highest resolution
def download_videos_from_file(file_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(file_path, 'r') as file:
        video_urls = file.read().splitlines()
        for url in video_urls:
            download_highest_resolution_video(url, output_directory)

input_file = "/content/gdrive/MyDrive/YT_LINKS.txt"  # Change this to your file containing video URLs
output_directory = "/content/gdrive/MyDrive/Wav2Lip"  # Change this to your desired output directory
download_videos_from_file(input_file, output_directory)

!git clone https://github.com/Rudrabha/Wav2Lip.git

#Replace the function below with _build_mel_basis() in /content/Wav2Lip/audio.py


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
    )

# !git clone https://github.com/Rudrabha/Wav2Lip.git

!ls /content/gdrive/MyDrive/Wav2Lip

!cp -ri "/content/gdrive/MyDrive/Wav2lip/wav2lip_gan.pth" /content/Wav2Lip/checkpoints/

!pip uninstall tensorflow tensorflow-gpu

!cd Wav2Lip && pip install -r requirements.txt

!wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "Wav2Lip/face_detection/detection/sfd/s3fd.pth"

!cp "/content/gdrive/My Drive/Wav2Lip/TechNews_Edit_2.mp4" "/content/gdrive/My Drive/Wav2Lip/output10.wav" sample_data/

!ls sample_data/

!cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/TechNews_Edit_2.mp4" --audio "../sample_data/output10.wav" --pads 0 20 0 0 --resize_factor 2 --nosmooth


# !cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/TechNews_1080.mp4" --audio "../sample_data/output10.wav"

!ls /content/gdrive/MyDrive/Wav2Lip/

# !cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/TechNews_Edit_2.mp4" --audio "../sample_data/output10.wav" --pads 0 20 0 0 --resize_factor 2 --nosmooth

# !cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/input_vid.mp4" --audio "../sample_data/input_audio.wav" --resize_factor 2



# !cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/TechNews_Edit_2.mp4" --audio "../sample_data/output10.wav" --nosmooth

