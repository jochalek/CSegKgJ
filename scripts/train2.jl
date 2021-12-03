using DrWatson
@quickactivate "CSegKgJ"

# using CSV, DataFrames
# df_train = CSV.read(Config.data_path * "train.csv", DataFrame);

using FastAI, Metalhead
using Wandb, FluxTraining, Logging
import CairoMakie; CairoMakie.activate!(type="png")

runname = CSegKgJ.runfileid()

# Wandb logging
lgbackend = WandbBackend(project = "CSegKgJ1",
                        name = "CSegKgJ-$runname",
                        "learning_rate" => 1.20226e-7,
                        "Projective Transforms" => (576, 704),
                        "batchsize" => 1,
                        "backbone" => "ResNet50",
                        "dataset" => "Kaggle sartorius-cell-instance-segmentation")


metricscb = LogMetrics(lgbackend)
hparamscb = LogHyperParams(lgbackend)

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
        ProjectiveTransforms(get_config(lgbackend, "Projective Transforms")),
        ImagePreprocessing(),
        OneHot()
    )
)

backbone = Metalhead.ResNet50().layers[1][1:end-1]
model = methodmodel(method, backbone);
lossfn = methodlossfn(method)


traindl, validdl = methoddataloaders(data, method, get_config(lgbackend, "batchsize"))
optimizer = ADAM()
learner = Learner(model, (traindl, validdl), optimizer, lossfn, ToGPU(), metricscb,  hparamscb)

fitonecycle!(learner, 30, 0.630957)

savemethodmodel(projectdir() * "/models/" * "$runname" * ".jld2", method, learner.model, force = true)