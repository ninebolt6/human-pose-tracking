import configparser
from dataclasses import dataclass
from typing import TypedDict

config = configparser.ConfigParser()
config.read("config/config.ini")


@dataclass(frozen=True)
class CommonConfig:
    OutputPath: str


@dataclass(frozen=True)
class TrackConfig(CommonConfig):
    InputPath: str
    ModelName: str
    OutputEnabled: bool
    ShowPreview: bool


@dataclass(frozen=True)
class ConvertConfig(CommonConfig):
    InputPath: str


def get_track_config() -> TrackConfig:
    return TrackConfig(
        OutputPath=config["common"]["OutputPath"],
        InputPath=config["track"]["InputPath"],
        ModelName=config["track"]["ModelName"],
        OutputEnabled=config["track"].getboolean("OutputEnabled"),
        ShowPreview=config["track"].getboolean("ShowPreview"),
    )


def get_convert_config():
    return ConvertConfig(
        OutputPath=config["common"]["OutputPath"],
        InputPath=config["convert"]["InputPath"],
    )
