# cython: language_level=3, boundscheck=False

from cpython cimport array
cimport numpy
cimport cython

#@cython.language_level("3")
cdef addOnlineObservationsLag1(
        numpy.ndarray[numpy.uint8_t, ndim=1] observations,
        numpy.uint8_t startObservationTime,
        numpy.ndarray[numpy.uint8_t, ndim=1] sourceNextPastCount,
        numpy.ndarray[numpy.uint8_t, ndim=1] sourcePastCount,
        numpy.ndarray[numpy.uint8_t, ndim=1] nextPastCount,
        numpy.ndarray[numpy.uint8_t, ndim=1] pastCount,
        numpy.ndarray[numpy.uint8_t, ndim=1] nextCount,
        numpy.ndarray[numpy.uint8_t, ndim=1] source,
        numpy.ndarray[numpy.uint8_t, ndim=1] dest,
        numpy.uint8_t maxShiftedValue,
        numpy.uint8_t maxShiftedSourceValue,
        numpy.uint8_t sourceEmbeddingDelay = 1,
        numpy.uint8_t destEmbeddingDelay = 1,
        numpy.uint8_t sourceHistoryEmbedLength = 1,
        numpy.uint8_t k = 1, #history
        numpy.uint8_t delay = 1,
        numpy.uint8_t base = 2,
        numpy.uint8_t startTime=0,
        numpy.uint8_t endTime=0
        ):

    if endTime == 0:
        endTime = len(dest) - 1

    if ((endTime - startTime) - startObservationTime + 1 <= 0):
        # No observations to add
        return

    if (endTime >= len(dest) or endTime >= len(source)):
        msg = "endTime {:d} must be <= length of input arrays (dest: {:d}, source: {:d})".format(endTime,
                                                                                                 dest.shape[0],
                                                                                                 source.shape[0])
        raise RuntimeError(msg)

    observations += (endTime - startTime) - startObservationTime + 1

    # Initialise and store the current previous values;
    #  one for each phase of the embedding delay.
    # First for the destination:
    cdef array.array pastVal = array.array(shape=destEmbeddingDelay, dtype=int)
    pastVal[0] = 0

    cdef array.array sourcePastVal = array.array(shape=sourceEmbeddingDelay, dtype=int)

    sourcePastVal[0] = 0

    destVal = 0
    destEmbeddingPhase = 0
    sourceEmbeddingPhase = 0
    startIndex = startTime + startObservationTime
    endIndex = endTime + 1
    for r in list(range(startIndex, endIndex)):
        if k > 0:
            pastVal[destEmbeddingPhase] += dest[r - 1]
        sourcePastVal[sourceEmbeddingPhase] += source[r - delay]
        # Add to the count for this particular transition
        # (cell's assigned as above
        destVal = dest[r]
        thisPastVal = pastVal[destEmbeddingPhase]
        thisSourceVal = sourcePastVal[sourceEmbeddingPhase]
        sourceNextPastCount[thisSourceVal][destVal][thisPastVal] += 1
        sourcePastCount[thisSourceVal][thisPastVal] += 1
        nextPastCount[destVal][thisPastVal] += 1
        pastCount[thisPastVal] += 1
        nextCount[destVal] += 1
        if k > 0:
            pastVal[destEmbeddingPhase] -= maxShiftedValue[dest[r - 1 - (k - 1) * destEmbeddingDelay]]
            pastVal[destEmbeddingPhase] *= base

        sourcePastVal[sourceEmbeddingPhase] -= maxShiftedSourceValue[
            source[r - delay - (sourceHistoryEmbedLength - 1) * sourceEmbeddingDelay]]
        sourcePastVal[sourceEmbeddingPhase] *= base
        # then update the phase
        destEmbeddingPhase = (destEmbeddingPhase + 1) % destEmbeddingDelay
        sourceEmbeddingPhase = (sourceEmbeddingPhase + 1) % sourceEmbeddingDelay


