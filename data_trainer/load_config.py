import configparser


def custom_roi(roi):
    temp = roi.split(",")
    return [(int(shape.split("/")[0]), int(shape.split("/")[1])) for shape in temp]


def config_set():
    config = configparser.ConfigParser()
    config.read("config.ini")

    config_dict = {}

    try:
        # trainer
        source_path = config["trainer"]["source_path"]
        if "\\" in source_path:
            source_path = source_path.replace("\\", "/")

        config_dict["source_path"] = source_path
        config_dict["categories"] = [cat.strip() for cat in config["trainer"]["categories"].split(",")]
        config_dict["validation_ratio"] = float(config["trainer"]["validation_ratio"])
        config_dict["sample_image"] = int(config["trainer"]["sample_image"])
        config_dict["epoch"] = int(config["trainer"]["epoch"])
        config_dict["batch_size"] = int(config["trainer"]["batch_size"])

        # roi
        config_dict["setting_roi"] = int(config["roi"]["setting_roi"])
        config_dict["custom_roi"] = int(config["roi"]["custom_roi"])

        config_dict["set_roi"] = config["roi"]["set_roi"]
        config_dict["cus_roi"] = custom_roi(config["roi"]["cus_roi"])

        # image_size
        config_dict["height"] = int(config["image_size"]["height"])
        config_dict["width"] = int(config["image_size"]["width"])
        config_dict["depth"] = int(config["image_size"]["depth"])

        # model
        config_dict["base_model"] = int(config["model"]["base_model"])
        config_dict["custom_model"] = int(config["model"]["custom_model"])

    except Exception as err:
        print(err)

    return config_dict


if __name__ == "__main__":
    print(config_set())
