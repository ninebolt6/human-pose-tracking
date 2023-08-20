import configparser
from dataclasses import dataclass
import json

config = configparser.ConfigParser()
config.read("config/config.ini")


@dataclass(frozen=True)
class CommonConfig:
    OutputPath: str
    SourcePoints: list[tuple[int, int]]
    DestinationSize: tuple[int, int]


@dataclass(frozen=True)
class TrackConfig(CommonConfig):
    InputPath: str
    ModelName: str
    OutputEnabled: bool
    ShowPreview: bool


@dataclass(frozen=True)
class ConvertConfig(CommonConfig):
    InputPath: str
    CalcInterval: int


def get_common_config() -> CommonConfig:
    return CommonConfig(
        OutputPath=config["common"]["OutputPath"],
        SourcePoints=json.loads(config["common"]["SourcePoints"]),
        DestinationSize=tuple(map(int, config["common"]["DestinationSize"].strip("()").split(","))),
    )


def get_track_config() -> TrackConfig:
    common_config = get_common_config()

    return TrackConfig(
        OutputPath=common_config.OutputPath,
        SourcePoints=common_config.SourcePoints,
        DestinationSize=common_config.DestinationSize,
        InputPath=config["track"]["InputPath"],
        ModelName=config["track"]["ModelName"],
        OutputEnabled=config["track"].getboolean("OutputEnabled"),
        ShowPreview=config["track"].getboolean("ShowPreview"),
    )


def get_convert_config():
    common_config = get_common_config()

    return ConvertConfig(
        OutputPath=common_config.OutputPath,
        SourcePoints=common_config.SourcePoints,
        DestinationSize=common_config.DestinationSize,
        InputPath=config["convert"]["InputPath"],
        CalcInterval=config["convert"].getint("CalcInterval"),
    )
