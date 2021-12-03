

function decode_rle(a, mask)
    # a = df[1, [:annotation]]
    # mask = zeros(UInt8, 520*704, 1) # FIXME hardcoded index into the df for dims
    s = split(a, " ")
    z = zeros(Int, length(s))
    for i in 1:length(s)
        z[i]=parse(Int, s[i])
    end
    for i in range(1, length(z)÷2, step=2)
        for k in 0:z[i+1]-1
            mask[z[i]+k] = 1
        end
    end
    # mask = reshape(mask, 520, :)
    return mask
end


using FileIO, ImageIO, DataFrames

function gen_masks(df)
    if isdir("../data/masks")
    else mkdir("../data/masks")
    end

    for i in DataFrames.unique(df[!, :id])
        df_filter = DataFrames.filter(:id => ==(i), df)
        mask = zeros(UInt8, 520*704) # hardcoded dims
        for j in 1:length(df_filter[:, 1])
            CSegKgJ.decode_rle(df_filter[j, :annotation], mask)
        end
        filename = i
        if isfile("../data/masks/" * "$filename" * ".png")
            println("not overwriting $filename")
            break
        else
            mask = permutedims(reshape(mask, 704, :))
            save(File{format"PNG"}("../data/masks/" * "$filename" * ".png"), mask)
        end
    end
end

using Dates: now

# Generate unique names for saving models
function runfileid()
    time = now()
    name = replace("$time", ":" => ".")
    return name
end