#!/bin/bash

MAP_TEST_URL="https://bit.ly/viper_test_map"
MAP_TRAIN_URL="https://bit.ly/viper_train_map"
MODEL_URL=""
MAP_TEST="maps_test.zip"
MAP_TRAIN="maps_train.zip"

if command -v wget > /dev/null; then
    echo "Downloading maps_test.zip"
    wget -O "$MAP_TEST" "$MAP_TEST_URL"

    echo "Downloading maps_train.zip"
    wget -O "$MAP_TRAIN" "$MAP_TRAIN_URL"

    echo "Downloading ViPER model"
    wget -O "$MODEL_URL"
else
    echo "Error: wget is not installed."
    exit 1
fi


if command -v unzip > /dev/null; then
    unzip "$MAP_TEST"
    unzip "$MAP_TRAIN"
    rm "$MAP_TEST"
    rm "$MAP_TRAIN"
else
    echo "Error: unzip is not installed."
    exit 1
fi

