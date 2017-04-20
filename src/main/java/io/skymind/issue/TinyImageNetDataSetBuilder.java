package io.skymind.issue;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.*;
import java.net.URI;
import java.net.URL;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Created by kepricon on 17. 4. 1.
 */
@Slf4j
public class TinyImageNetDataSetBuilder {
    private static final String DATA_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip";
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "tiny-imagenet");
    private static final String DATA_ROOT_DIR = FilenameUtils.concat(DATA_PATH, "tiny-imagenet-200/");
    private static final String LABEL_ID_FILE = DATA_ROOT_DIR + "wnids.txt";
    private static final String LABEL_NAME_FILE = DATA_ROOT_DIR + "words.txt";
    private static final String TRAIN_DIR = DATA_ROOT_DIR + "train/";
    private static final String VALIDATION_DIR = DATA_ROOT_DIR + "val/";
    private static final String VALIDATION_ANNOTATION_FILE = DATA_ROOT_DIR + "val/val_annotations.txt";
    private static final String[] allowedExtensions = new String[]{"JPEG"};

    @Parameter(names = {"-w","--width"}, description = "WIDTH_SIZE")
    private static int width = 224;
    @Parameter(names = {"-h","--height"}, description = "HEIGHT_SIZE")
    private static int height = 224;
    @Parameter(names = {"-c","--channel"}, description = "CHANNEL_SIZE")
    private static int channel = 3;
    @Parameter(names = {"-b","--batch"}, description = "BATCH_SIZE")
    private static int batchSize = 16;
    @Parameter(names = {"-m","--maxBatch"}, description = "MAX_EXAMPLE_BATCH_SIZE")
    private static int maxExampleBatches = 100;

    public static final String DATASETNAME = "tiny";
    public static final int numLabels = 200;

    public static String TRAIN_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tiny_32batch_160/dl4j_tinyimagenet_train/");
    public static String TEST_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tiny_32batch_160/dl4j_tinyimagenet_test/");

    public void run(String[] args) throws Exception {
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            System.exit(1);
        }

        downloadData();

        TRAIN_PATH = getTrainPath(height, width, batchSize);
        TEST_PATH = getTestPath(height, width, batchSize);

        if(new File(TRAIN_PATH).exists() == false){

            Random r = new Random(12345);
            FileSplit trainSplit = new FileSplit(new File(TRAIN_DIR), allowedExtensions, r);
            FileSplit valSplit = new FileSplit(new File(VALIDATION_DIR), allowedExtensions, r);

            List<String> labelIDs = loadLabels(LABEL_ID_FILE);
            List<String> labelNames = loadLabelNames(labelIDs, LABEL_NAME_FILE);

            log.info("Loading data...");
            ImageRecordReader trainReader = new ImageRecordReader(height,width,channel,new TrainLabelGenerator(labelIDs));
            trainReader.initialize(trainSplit);
            trainReader.setLabels(labelIDs);
            ImageRecordReader testReader = new ImageRecordReader(height,width,channel, new ValidationLabelGenerator(labelIDs, VALIDATION_ANNOTATION_FILE));
            testReader.initialize(valSplit);
            testReader.setLabels(labelIDs);


            DataSetIterator trainData = new RecordReaderDataSetIterator(trainReader, batchSize, 1, numLabels);
            DataSetIterator testData = new RecordReaderDataSetIterator(testReader, batchSize, 1, numLabels);
            trainData.setPreProcessor(new ImagePreProcessingScaler(-1,1,8));
            testData.setPreProcessor(new ImagePreProcessingScaler(-1,1,8));

            log.info("create train datasets in " + TRAIN_PATH);
            new File(TRAIN_PATH).mkdirs();

            AtomicInteger counter = new AtomicInteger(0);
            while(trainData.hasNext()){
                if (counter.intValue() > maxExampleBatches) {
                    break;
                }
                String path = FilenameUtils.concat(TRAIN_PATH, "dataset-" + (counter.getAndIncrement()) + ".bin");
                trainData.next().save(new File(path));

                if (counter.get() % 100 == 0) {
                    log.info("{} datasets saved so far...", counter.get());
                }
            }

//            log.info("create test datasets in " + TEST_PATH);
//            new File(TEST_PATH).mkdirs();
//
//            counter = new AtomicInteger(0);
//            while(testData.hasNext()){
//                String path = FilenameUtils.concat(TEST_PATH, "dataset-" +  (counter.getAndIncrement()) + ".bin");
//                testData.next().save(new File(path));
//
//                if (counter.get() % 100 == 0) {
//                    log.info("{} datasets saved so far...", counter.get());
//                }
//            }

        }
    }

    public static String getTrainPath(int height, int width, int batchSize){
        StringBuffer sb = new StringBuffer();
        sb.append("dl4j_");
        sb.append(DATASETNAME + "_");
        sb.append(batchSize + "batch_");
        sb.append(width+"x"+height);
        sb.append("/dl4j_tinyimagenet_train/");

        return FilenameUtils.concat(System.getProperty("java.io.tmpdir"), sb.toString());
    }

    public static String getTestPath(int height, int width, int batchSize){
        StringBuffer sb = new StringBuffer();
        sb.append("dl4j_");
        sb.append(DATASETNAME + "_");
        sb.append(batchSize + "batch_");
        sb.append(width+"x"+height);
        sb.append("/dl4j_tinyimagenet_test/");

        return FilenameUtils.concat(System.getProperty("java.io.tmpdir"), sb.toString());
    }

    private static void downloadData() throws Exception{

        File directory = new File(DATA_PATH);
        if (false == directory.exists()) {
            directory.mkdirs();
        }

        File zipFile = new File(DATA_PATH, "tiny-imagenet-200.zip");

        if (false == zipFile.exists()){
            log.info("Starting data download (248MB) ...");
            FileUtils.copyURLToFile(new URL(DATA_URL), zipFile);
            log.info("Data (.zip file) downloaded to " + zipFile.getAbsolutePath());
            extractZip(zipFile.getAbsolutePath(), DATA_PATH);
        }
    }


    private static final int BUFFER_SIZE = 4096;
    private static void extractZip(String filePath, String outputPath) throws IOException {
        int fileCount = 0;
        int dirCount = 0;
        log.info("Extracting files");

        try(ZipInputStream zis = new ZipInputStream(new FileInputStream(filePath))){
            ZipEntry entry;

            while ((entry = zis.getNextEntry()) != null){
                if(entry.isDirectory()){
                    new File(outputPath, entry.getName()).mkdirs();
                    dirCount++;
                }else {
                    int count;
                    byte data[] = new byte[BUFFER_SIZE];

                    FileOutputStream fos = new FileOutputStream(new File(outputPath, entry.getName()));
                    BufferedOutputStream dest = new BufferedOutputStream(fos,BUFFER_SIZE);
                    while ((count = zis.read(data, 0, BUFFER_SIZE)) != -1) {
                        dest.write(data, 0, count);
                    }
                    dest.close();
                    fileCount++;
                }
            }
        }
        log.info("\n" + fileCount + " files and " + dirCount + " directories extracted to: " + outputPath);
    }

    private static List<String> loadLabels(String path) throws IOException {
        List<String> lines = FileUtils.readLines(new File(path));
        List<String> out = new ArrayList<>(200);
        for(String s : lines){
            if(s.length() > 0){
                out.add(s);
            }
        }
        return out;
    }

    private static List<String> loadLabelNames(List<String> labelIDs, String path ) throws IOException {
        Map<String,String> indexesToNames = new HashMap<>();
        List<String> lines = FileUtils.readLines(new File(path));
        for(String s : lines){
            String[] split = s.split("\t");
            indexesToNames.put(split[0],split[1]);
        }

        List<String> out = new ArrayList<>(labelIDs.size());
        for(String s : labelIDs){
            out.add(indexesToNames.get(s));
        }
        return out;
    }

    private static class TrainLabelGenerator implements PathLabelGenerator {
        private Map<String,Integer> labelIdxs;

        public TrainLabelGenerator(List<String> labels) throws IOException {
            labelIdxs = new HashMap<>();
            int i=0;
            for(String s : labels){
                labelIdxs.put(s, i++);
            }
        }

        @Override
        public Writable getLabelForPath(String path) {
            String dirName = FilenameUtils.getBaseName(new File(path).getParentFile().getParent());
            return new Text(dirName);
        }

        @Override
        public Writable getLabelForPath(URI uri) {
            return getLabelForPath(uri.toString());
        }
    }

    private static class ValidationLabelGenerator implements PathLabelGenerator {
        private Map<String,Integer> labelIdxs;
        private Map<String,String> filenameToIndex;

        private ValidationLabelGenerator(List<String> labels, String annotationsFile) throws IOException {
            labelIdxs = new HashMap<>();
            int i=0;
            for(String s : labels){
                labelIdxs.put(s, i++);
            }
            this.filenameToIndex = loadValidationSetLabels(annotationsFile);
        }

        @Override
        public Writable getLabelForPath(String path) {
            File f = new File(path);
            String filename = f.getName();
            return new Text(filenameToIndex.get(filename));
        }

        @Override
        public Writable getLabelForPath(URI uri) {
            return getLabelForPath(uri.toString());
        }
    }

    private static Map<String,String> loadValidationSetLabels(String path) throws IOException {
        Map<String,String> validation = new HashMap<>();
        List<String> lines = FileUtils.readLines(new File(path));
        for(String s : lines){
            String[] split = s.split("\t");
            validation.put(split[0],split[1]);
        }
        return validation;
    }

    public static void main(String[] args) throws Exception {
        new TinyImageNetDataSetBuilder().run(args);
    }
}

