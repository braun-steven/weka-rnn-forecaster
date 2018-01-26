package weka.classifiers.functions;

import weka.classifiers.timeseries.core.StateDependentPredictor;
import weka.core.Instances;

/**
 * A forecaster class using recurrent units.
 *
 * @author Steven Lang
 */
public class RnnForecaster extends RnnSequenceClassifier implements StateDependentPredictor {

    /**
     * SerialVersionUID
     */
    private static final long serialVersionUID = -7108781255562512907L;

    @Override
    public void initializeClassifier(Instances data) throws Exception {
        super.initializeClassifier(data);
    }

    @Override
    protected void createModel() throws Exception {
        super.createModel();
    }

    @Override
    public double[][] distributionsForInstances(Instances insts) throws Exception {
        return super.distributionsForInstances(insts);
    }

    @Override
    public void clearPreviousState() {

    }

    @Override
    public void setPreviousState(Object previousState) {

    }

    @Override
    public Object getPreviousState() {
        return null;
    }

    @Override
    public void serializeState(String path) throws Exception {

    }

    @Override
    public void loadSerializedState(String path) throws Exception {

    }
}
