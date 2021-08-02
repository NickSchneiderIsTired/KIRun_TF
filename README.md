# Installation

## Pip
In order to install required packages, you need to install `pip`.
Install required packages from `requirements.txt` by running `py -m pip install -r requirements.txt`

## SoX
You need to manually install [SoX](http://sox.sourceforge.net/) to get OpenSMILE to work properly. Make sure the installation directory is set in your PATH variable

# Training

## Data directories

Training and validation data need to put into the folders `data/train/` and `data/val/`

## File formatting

Training files should consist of audio files and according groudtruth files

## Audio files

File type can be one of
- RIFF-WAVE (uncompressed PCM only)
- Comma Separated Value (CSV)
- HTK parameter files
- WEKA’s ARFF format

## Groundtruth

Groundtruth files of according audio files should have the exact same filename, but the filetype
The GT filetype is `.txt` and should be formatted like the following:

*startTimeStamp* *endTimeStamp* aw_[-5, 5]
*startTimeStamp* *endTimeStamp* ab_[6, 20]
*startTimeStamp* *endTimeStamp* au_[groundtype]
...

aw=answer wealth, ab=answer burden, au=answer groundtype.
There can be an infinite amount of this set of 3 lines

# Usage

The already trained model is located under `trainedModel/`.
To use it, and test a runners burden use the following command:
`python checkburden --file path/to/audio.wav [--timestamp *startTimeStampInSeconds*]`
- `file`:  Path to audio file of the runners sample
- `timestamp (optional)`: Timestamp to start reading 5 seconds of data at. (Default: 0)

