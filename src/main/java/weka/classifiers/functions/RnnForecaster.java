package weka.classifiers.functions;

import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import weka.classifiers.timeseries.core.BaseModelSerializer;
import weka.classifiers.timeseries.core.StateDependentPredictor;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.MissingOutputLayerException;
import weka.dl4j.CacheMode;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * A forecaster class using recurrent units.
 *
 * @author Steven Lang
 * @author pedrofale
 */
public class RnnForecaster extends RnnSequenceClassifier implements StateDependentPredictor, BaseModelSerializer {

    /**
     * SerialVersionUID
     */
    private static final long serialVersionUID = -7108781255562512907L;

    @Override
    public void initializeClassifier(Instances data) throws Exception {
        // Can classifier handle the data?
        getCapabilities().testWithFail(data);

        // Check basic network structure
        if (layers.length == 0) {
            throw new MissingOutputLayerException("No layers have been added!");
        }

        final Layer lastLayer = layers[layers.length - 1];
        if (!(lastLayer instanceof RnnOutputLayer)) {
            throw new MissingOutputLayerException("Last layer in network must be an output layer!");
        }

        ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
        try {
            Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());

            data = initEarlyStopping(data);
            this.trainData = data;

            instanceIterator.initialize();

            createModel();

            // Setup the datasetiterators (needs to be done after the model initialization)
            trainIterator = getDataSetIterator(this.trainData);

            // Set the iteration listener
            model.setListeners(getListener());

            numEpochsPerformed = 0;
        } finally {
            Thread.currentThread().setContextClassLoader(origLoader);
        }
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enableAll();

        return result;
    }

    @Override
    protected void createModel() throws Exception {
        super.createModel();
    }



    @Override
    public void buildClassifier(Instances data) throws Exception {
        super.buildClassifier(data);
    }

    @Override
    public double[][] distributionsForInstances(Instances insts) throws Exception {
        if (zeroR != null) {
            return zeroR.distributionsForInstances(insts);
        }

        // Apply filter
        applyFilters(insts);

        // Get predictions
        final DataSetIterator it = getDataSetIterator(insts, CacheMode.NONE);
        double[][] preds = new double[insts.numInstances()][insts.numClasses()];

        if (it.resetSupported()) {
            it.reset();
        }

        int offset = 0;
        boolean next = it.hasNext();

        // Get predictions batch-wise
        while (next) {
            final DataSet ds = it.next();
            final INDArray features = ds.getFeatureMatrix();
            final INDArray labelsMask = ds.getLabelsMaskArray();
            INDArray lastTimeStepIndices = Nd4j.argMax(labelsMask, 1);
            INDArray predBatch = model.rnnTimeStep(features)[0];
            int currentBatchSize = predBatch.size(0);
            for (int i = 0; i < currentBatchSize; i++) {
                int thisTimeSeriesLastIndex = lastTimeStepIndices.getInt(i);
                INDArray thisExampleProbabilities =
                        predBatch.get(
                                point(i),
                                all(),
                                point(thisTimeSeriesLastIndex));
                for (int j = 0; j < insts.numClasses(); j++) {
                    preds[i + offset][j] = thisExampleProbabilities.getDouble(j);
                }
            }

            offset += currentBatchSize; // add batchsize as offset
            boolean hasInstancesLeft = offset < insts.numInstances();
            next = it.hasNext() || hasInstancesLeft;
        }






//        INDArray testMatrix = testData.getFeatureMatrix();

        // For RNNs the state of the network is important (namely all the LSTM layers). After training it with the
        // given data, the prediction it makes for the next time step must take all the past into account. Of course
        // it would be rather unefficient for it to go all the way back to the beginning of the time series each time
        // made a prediction, so we use rnnTimeStep to grab the current network state (which knows what the last output
        // was) and make the next prediction only based on the last time step.

//        INDArray predicted = model.rnnTimeStep(testMatrix);

        // using output() would require us to input to the test matrix all the past instances
        // with rnnTimeStep() we can just input the last instance, as it has stored the state from the last prediction
        // so conceptually all the past data is still always used

//        predicted = predicted.getRow(0);
//        double[] preds = new double[inst.numClasses()];
        for (int i = 0; i < preds.length; i++) {
//            preds[i] = predicted.getDouble(i);
            weka.core.Utils.normalize(preds[i]);
        }
        return preds;
    }


    @Override
    public void clearPreviousState() {
        model.rnnClearPreviousState();
    }

    @Override
    public void setPreviousState(Object previousState) {
        List<Map<String, INDArray>> states = (List<Map<String, INDArray>>) previousState;
        int numLayers = layers.length;
        for (int i = 0; i < numLayers; i++) {
            if (layers[i] instanceof LSTM)
                model.rnnSetPreviousState(i, states.get(i));
        }
    }

    /**
     * Get the previous state of each LSTM layer, as seen by a StateDependentPredictor.
     *
     * @return list of state for each layer
     */
    @Override
    public List<Map<String, INDArray>> getPreviousState() {
        List<Map<String, INDArray>> states = new ArrayList<>(getNumLSTMLayers());
        int numLayers = layers.length;

        for (int i = 0; i < numLayers; i++) {
            if (isStatefulLayer(layers[i]))
                states.add(model.rnnGetPreviousState(i));
        }
        return states;
    }

    /**
     * Check if a layer is stateful.
     * <p>
     * TODO: Can be static
     *
     * @param layer Layer to be checked
     * @return True if layer is instance of a stateful layer class
     */
    protected boolean isStatefulLayer(Layer layer) {
        return layer instanceof LSTM
                || layer instanceof GravesLSTM;
    }

    @Override
    public void serializeState(String path) throws Exception {
        File sFile = new File(path);
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(sFile));
        List<Map<String, INDArray>> states = getPreviousState();
        oos.writeObject(states);
        oos.close();
    }

    @Override
    public void loadSerializedState(String path) throws Exception {
        File sFile = new File(path);
        ObjectInputStream is = new ObjectInputStream(new FileInputStream(sFile));
        Object states = is.readObject();
        is.close();
        setPreviousState(states);
    }

    @Override
    public void serializeModel(String path) throws Exception {
        File file = new File(path);
        ModelSerializer.writeModel(model, file, true);
    }

    @Override
    public void loadSerializedModel(String path) throws Exception {
        File sFile = new File(path);
        model = ModelSerializer.restoreComputationGraph(sFile);

    }

    /**
     * Get the number of LSTM layers
     *
     * @return number of LSTM layers
     */
    private int getNumLSTMLayers() {
        return (int) Arrays.stream(layers)
                .filter(this::isStatefulLayer)
                .count();
    }

    /**
     * Get the indexes of the LSTM layers of the RNN
     *
     * @return a list containing the indexes
     */
    private List<Integer> getLSTMindexes() {
        List<Integer> indexes = new ArrayList<>(getNumLSTMLayers());
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof LSTM)
                indexes.add(i);
        }
        return indexes;
    }
}
