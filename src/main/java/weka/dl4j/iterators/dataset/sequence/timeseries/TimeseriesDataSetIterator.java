package weka.dl4j.iterators.dataset.sequence.timeseries;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.iterators.dataset.DefaultDataSetIterator;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class TimeseriesDataSetIterator implements DataSetIterator {
    private static final long serialVersionUID = -8938690401284676734L;

    /** Dataset */
    protected final Instances data;
    /** Batch size */
    protected final int batchSize;
    /** Cursor to current row */
    protected int cursor;
    /** Maximum sequence length */
    protected final int truncateLength;

    /**
     * Constructor.
     *
     * @param data Dataset
     * @param batchSize Batch size
     * @param truncateLength Maximum sequence length
     */
    public TimeseriesDataSetIterator(
            Instances data, int batchSize, int truncateLength) {
        this.data = data;
        this.batchSize = batchSize;
        this.cursor = 0;
        this.truncateLength = truncateLength;
    }

    /**
     * Converts a set of training instances corresponding to a time series to a DataSet.
     * For RNNs, the input of the network is 3 dimensional of dimensions [numExamples, numInputs, numTimeSteps]
     * Assumes that the instances have been suitably preprocessed - i.e. missing values replaced
     * and nominals converted to binary/numeric. Also assumes that the class index has been set
     *
     * @param num Next batch size
     * @return a DataSet
     */
    @Override
    public DataSet next(int num) {
        List<Instance> batch = new ArrayList<>(num);

        for (int i = 0; i < num && cursor + i < data.numInstances(); i++) {
            batch.add(data.get(cursor + i));
        }

        final int currentBatchSize = batch.size();
        INDArray features = Nd4j.ones(1, data.numAttributes() - 1, currentBatchSize);
        INDArray labels = Nd4j.ones(1, data.numClasses(), currentBatchSize);
        double[] outcomes = new double[currentBatchSize];

        for (int i = 0; i < currentBatchSize; i++) {
            double[] independent = new double[data.numAttributes() - 1];
            int index = 0;
            Instance current = data.instance(cursor + i);
            for (int j = 0; j < data.numAttributes(); j++) {
                if (j != data.classIndex()) {
                    independent[index++] = current.value(j);
                } else {
                    outcomes[i] = current.classValue();
                }
            }

            INDArray ind = Nd4j.create(independent, new int[]{1, data.numAttributes() - 1, 1});
            for(int k = 0; k < independent.length; k++) {
                features.putScalar(0, k, i, ind.getDouble(k));
            }
        }
        INDArray outcomesNDArray = Nd4j.create(outcomes, new int[] {1, currentBatchSize, 1});
        labels.putColumn(0, outcomesNDArray);

        DataSet dataSet = new DataSet(features, labels);
        cursor += currentBatchSize;
        return dataSet;

    }

    @Override
    public int totalExamples() {
        return data.numInstances();
    }

    @Override
    public int inputColumns() {
        return data.numAttributes();
    }

    @Override
    public int totalOutcomes() {
        return data.numClasses();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return data.numInstances();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return data.stream()
                .map(Instance::classValue)
                .map(String::valueOf)
                .distinct()
                .sorted()
                .collect(Collectors.toList());
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
