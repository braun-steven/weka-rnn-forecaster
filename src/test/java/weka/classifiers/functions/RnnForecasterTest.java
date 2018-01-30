package weka.classifiers.functions;

import org.junit.Assert;
import org.junit.Test;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.dl4j.activations.ActivationTanH;
import weka.dl4j.iterators.instance.sequence.timeseries.TimeseriesInstanceIterator;
import weka.dl4j.layers.LSTM;
import weka.dl4j.layers.RnnOutputLayer;
import weka.dl4j.lossfunctions.LossMSE;
import weka.filters.supervised.attribute.TSLagMaker;

import java.util.List;

/**
 * TestCase for {@link RnnForecaster}.
 *
 * @author pedrofale
 * @author Steven Lang
 */
public class RnnForecasterTest {

    private String predsToString(List<List<NumericPrediction>> preds, int steps) {
        StringBuilder b = new StringBuilder();

        for (int i = 0; i < steps; i++) {
            List<NumericPrediction> predsForTargetsAtStep =
                    preds.get(i);

            for (NumericPrediction p : predsForTargetsAtStep) {
                double[][] limits = p.predictionIntervals();
                b.append(p.predicted()).append(" ");
                if (limits != null && limits.length > 0) {
                    b.append(limits[0][0]).append(" ").append(limits[0][1]).append(" ");
                }
            }
            b.append("\n");
        }

        return b.toString();
    }

    public Instances getWineData() throws Exception {
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource("datasets/wine_date.arff");
        return ds.getDataSet();
    }

    public Classifier configureRNN() {
        RnnForecaster cls = new RnnForecaster();
        LSTM lstm = new weka.dl4j.layers.LSTM();
        lstm.setActivationFn(new ActivationTanH());
        lstm.setNOut(10);
        RnnOutputLayer out = new RnnOutputLayer();
        out.setLossFn(new LossMSE());
        cls.setLayers(lstm, out);
        cls.setNumEpochs(10);

        TimeseriesInstanceIterator tsii = new TimeseriesInstanceIterator();
        cls.setInstanceIterator(tsii);

        return cls;
    }

    @Test
    public void testForecastTwoTargetsConfidenceIntervals() throws Exception {

        boolean success = false;
        Instances wine = getWineData();

        WekaForecaster forecaster = new WekaForecaster();
        TSLagMaker lagMaker = forecaster.getTSLagMaker();

        try {
            forecaster.setBaseForecaster(configureRNN());
            forecaster.setFieldsToForecast("Fortified,Dry-white");
            forecaster.setCalculateConfIntervalsForForecasts(12);
            lagMaker.setTimeStampField("Date");
            lagMaker.setMinLag(1);
            lagMaker.setMaxLag(12);
            lagMaker.setAddMonthOfYear(true);
            lagMaker.setAddQuarterOfYear(true);
            forecaster.buildForecaster(wine, System.out);
            forecaster.primeForecaster(wine);

            int numStepsToForecast = 12;
            List<List<NumericPrediction>> forecast =
                    forecaster.forecast(numStepsToForecast, System.out);

            String forecastString = predsToString(forecast, numStepsToForecast);
            success = true;
            System.out.println(forecastString);
        } catch (Exception ex) {
            ex.printStackTrace();
            String msg = ex.getMessage().toLowerCase();
            if (msg.contains("not in classpath")) {
                return;
            }
        }

        if (!success) {
            Assert.fail("Problem during regression testing: no successful predictions generated");
        }
    }
}