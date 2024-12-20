from comet_ml import start
from comet_ml.integration.pytorch import log_model

if __name__ == "__main__":

    experiment = start(
      api_key="71buxqNAdfPLVBFw4MsusCH6h",
      project_name="w2v",
      workspace="dwalker93"
    )
