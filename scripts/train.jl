using DrWatson
@quickactivate "CSegKgJ"

# using CSV, DataFrames
# df_train = CSV.read(Config.data_path * "train.csv", DataFrame);

using FastAI, Metalhead
using Wandb, FluxTraining, Logging
import CairoMakie; CairoMakie.activate!(type="png")

# Wandb logging
backend = WandbBackend(project = "CSegKgJ1", name = nothing) # FIXME for random name
metricscb = LogMetrics(backend)
hparamscb = LogHyperParams(backend)


dir = "../data"
classes = ["background", "cell"]
images = Datasets.loadfolderdata(
    joinpath(dir, "train"),
    filterfn=isimagefile,
    loadfn=loadfile)

masks = Datasets.loadfolderdata(
    joinpath(dir, "masks"),
    filterfn=isimagefile,
    loadfn=f -> loadmask(f, classes))

data = (images, masks)

method = BlockMethod(
    (Image{2}(), Mask{2}(classes)),
    (
        ProjectiveTransforms((128, 128)),
        ImagePreprocessing(),
        OneHot()
    )
)

backbone = Models.xresnet50()
model = methodmodel(method, backbone);
lossfn = methodlossfn(method)


traindl, validdl = methoddataloaders(data, method, 16)
optimizer = ADAM()
learner = Learner(model, (traindl, validdl), optimizer, lossfn, ToGPU(), metricscb,  hparamscb)

fitonecycle!(learner, 600, 2.51189e-5)

savemethodmodel("../models/test1.jld2", method, learner.model, force = true)