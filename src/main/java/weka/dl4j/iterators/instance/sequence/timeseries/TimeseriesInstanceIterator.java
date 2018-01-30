package weka.dl4j.iterators.instance.sequence.timeseries;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Instances;
import weka.core.InvalidInputDataException;
import weka.dl4j.iterators.dataset.sequence.timeseries.TimeseriesDataSetIterator;
import weka.dl4j.iterators.instance.sequence.AbstractSequenceInstanceIterator;

public class TimeseriesInstanceIterator extends AbstractSequenceInstanceIterator {
    private static final long serialVersionUID = 8786494689414456345L;

    @Override
    public DataSetIterator getDataSetIterator(Instances data, int seed, int batchSize) throws Exception {
        TimeseriesDataSetIterator tsdsi = new TimeseriesDataSetIterator(data, batchSize, 0);
        return tsdsi;
    }

    @Override
    public void validate(Instances data) throws InvalidInputDataException {

    }
}
