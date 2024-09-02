## Run the docker container

``` bash
docker compose run arm64v8
```

in the container make sure that `/dev/bus/usb` is accessible to the gnuradio user, otherwise the LimeSDR won't be visible.

``` bash
sudo chgrp -R gnuradio /dev/bus/usb
```

## Run gnuradio-companion

In the container, run `gnuradio-companion`

``` bash
gnuradio-companion
```

the current directory is mapped to `/opt/project` so browse to `/opt/project/gnuradio/save_data` and open `test.grc`. 

you may need to check the directory where the data is saved, it should be `/data` otherwise there may be some errors. 
