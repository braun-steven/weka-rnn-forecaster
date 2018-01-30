# Weka RNN Timeseries Forecasting

This package provides a timeseries forecasting model for weka using Recurrent Neural Networks. 

## Build Locally

To build this package locally, it is necessary to have the following two packages in your local maven repository installed

- `wekaDeeplearning4j`
- `timeseriesForecasting`

To get these, simply run the `prepare.sh` script:

```bash
$ ./prepare.sh -h
Usage: prepare.sh

Optional arguments:
   -v/--verbose            Enable verbose mode
   -b/--backend            Select specific backend 
                           Available: ( CPU GPU )
   -h/--help               Show this message
```

It will download both packages and install the maven artifact in the local repository. 