//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// G4PropagatorInField 
//
// Class description:
// 
// This class performs the navigation/propagation of a particle/track 
// in a magnetic field. The field is in general non-uniform.
// For the calculation of the path, it relies on the class G4ChordFinder.
// It utilises an ODE solver (with the Runge-Kutta method) to evolve the
// particle, and drives it until the particle has traveled a set distance
// or it enters a new volume.

// Author: John Apostolakis (CERN), 25 October 1996
// ---------------------------------------------------------------------------
#ifndef G4PropagatorInField_hh 
#define G4PropagatorInField_hh 1

#include "G4Types.hh"

#include <vector>

#include "G4FieldTrack.hh"
#include "G4FieldManager.hh"
#include "G4VIntersectionLocator.hh"

class G4ChordFinder; 

class G4Navigator;
class G4VPhysicalVolume;
class G4VCurvedTrajectoryFilter;

/**
 * @brief G4PropagatorInField performs the navigation/propagation of a
 * particle/track in a magnetic field. The field is in general non-uniform.
 * For the calculation of the path, it relies on the class G4ChordFinder.
 * It utilises an ODE solver (with the Runge-Kutta method) to evolve the
 * particle, and drives it until the particle has traveled a set distance
 * or it enters a new volume.
 */

class G4PropagatorInField
{
  public:

    /**
     * Constructor and Destructor.
     */
    G4PropagatorInField( G4Navigator* theNavigator, 
                         G4FieldManager* detectorFieldMgr,
                         G4VIntersectionLocator* vLocator = nullptr );
   ~G4PropagatorInField();

    /**
     * Computes the next geometric Step.
     *  @param[in,out] pFieldTrack Field track to be filled.
     *  @param[in] pCurrentProposedStepLength Current proposed step length.
     *  @param[in,out] pNewSafety New safety.
     *  @param[in] pPhysVol Pointer to the current volume.
     *  @param[in] canRelaxDeltaChord To enable relaxing delta-chord parameter.
     *  @returns Step length.
     */
    G4double ComputeStep( G4FieldTrack& pFieldTrack,
                          G4double pCurrentProposedStepLength,
                          G4double& pNewSafety, 
                          G4VPhysicalVolume* pPhysVol = nullptr,
                          G4bool canRelaxDeltaChord = false);

    /**
     * Returning the state after the Step.
     */
    inline G4ThreeVector EndPosition() const;       
    inline G4ThreeVector EndMomentumDir() const;
    inline G4bool        IsParticleLooping() const;

    /**
     * Returning the relative accuracy for the current Step.
     */
    inline G4double GetEpsilonStep() const;

    /**
     * Setting the relative accuracy for the current Step.
     * The ratio DeltaOneStep()/h_current_step.
     */
    inline void SetEpsilonStep(G4double newEps);

    /**
     * Sets (and returns) the correct field manager (global or local), 
     * if it exists.
     *  @note Should be called before ComputeStep is called;
     *        Currently, ComputeStep() will call it, if it has not been called.
     *  @param[in] pCurrentPhysVol Pointer to the current volume.
     *  @returns The pointer to the field manager.
     */
    G4FieldManager* FindAndSetFieldManager(G4VPhysicalVolume* pCurrentPhysVol);
 
    /**
     * Returning the pointer to the chord finder.
     */
    inline G4ChordFinder* GetChordFinder();

    /**
     * Verbosity control.
     */
    G4int SetVerboseLevel( G4int verbose );
    inline G4int GetVerboseLevel() const;
    inline G4int Verbose() const;

    /**
     * Enabling check mode for further diagnostics.
     */
    inline void CheckMode(G4bool mode);

    /**
     * Accessor/modifier for tracing key parts of ComputeStep().
     */
    inline void SetVerboseTrace( G4bool enable );
    inline G4bool GetVerboseTrace();
   
    /**
     * Accessor/modifier for controlling the maximum for the number of
     * substeps that a particle can take. Above this number it is signaled
     * as 'looping'.
     */
    inline G4int GetMaxLoopCount() const;
    inline void  SetMaxLoopCount( G4int new_max );

    /**
     * Print method, useful mostly for debugging.
     */
    void printStatus( const G4FieldTrack&      startFT,
                      const G4FieldTrack&      currentFT, 
                            G4double           requestStep, 
                            G4double           safety,
                            G4int              step, 
                            G4VPhysicalVolume* startVolume);

    /**
     * Accessor for retrieving the field track.
     */
    inline G4FieldTrack GetEndState() const;

    /**
     * Methods to control values for the global field manager.
     *  @deprecated The four methods below are now obsolescent but *for now*
     *     will work. They are being replaced by same-name methods in
     *     G4FieldManager, allowing the specialisation in different volumes. 
     */
    inline G4double GetMinimumEpsilonStep() const; // Min for relative accuracy
    inline void     SetMinimumEpsilonStep( G4double newEpsMin ); // of any step
    inline G4double GetMaximumEpsilonStep() const;
    inline void     SetMaximumEpsilonStep( G4double newEpsMax );

    /**
     * Methods to obtain / change the size of the largest step the method
     * will undertake. The Reset method uses the world volume's.
     */
    void     SetLargestAcceptableStep( G4double newBigDist );
    G4double GetLargestAcceptableStep();
    void     ResetLargestAcceptableStep();

    /**
     * Methods to control extra Multiplier parameter for limiting long steps.
     */
    G4double GetMaxStepSizeMultiplier();
    void     SetMaxStepSizeMultiplier(G4double vm);

    /**
     * Methods to Control minimum 'directional' distance in case of
     * too-large step.
     */
    G4double GetMinBigDistance();
    void     SetMinBigDistance(G4double val);

    /**
     * Sets the filter that examines & stores 'intermediate' 
     * curved trajectory points.
     *  @note Currently only position is stored.
     */
    void SetTrajectoryFilter(G4VCurvedTrajectoryFilter* filter);

    /**
     * Accesses the points which have passed by the filter.
     *   @note Responsibility for deleting the points lies with the client.
     *         This method MUST BE called exactly ONCE per step.
     */ 
    std::vector<G4ThreeVector>* GimmeTrajectoryVectorAndForgetIt() const;

    /**
     * Clears the State of this class and its current associates.
     *   @note The current field manager & chord finder will also be called.
     */
    void ClearPropagatorState();

    /**
     * Setter for global field manager. Updates the state.
     */
    inline void SetDetectorFieldManager( G4FieldManager* newGlobalFieldManager );
  
    /**
     * Toggles & views parameter for using safety to discard unneccesary calls
     * to the navigator (thus 'optimising' performance).
     */
    inline void   SetUseSafetyForOptimization( G4bool );
    inline G4bool GetUseSafetyForOptimization();

    /**
     * Intersects the chord from StartPointA to EndPointB and returns
     * whether an intersection occurred.
     *   @note Safety is changed!
     */
    inline G4bool IntersectChord( const G4ThreeVector& StartPointA,
                                  const G4ThreeVector& EndPointB,
                                        G4double&      NewSafety,
                                        G4double&      LinearStepLength,
                                        G4ThreeVector& IntersectionPoint);

    /**
     * Returns if it is the first step in the volume.
     */
    inline G4bool IsFirstStepInVolume();

    /**
     * Returns if it is the last step in the volume.
     */
    inline G4bool IsLastStepInVolume();

    /**
     * Initialises track flags.
     */
    inline void PrepareNewTrack();

    /**
     * Changes or gets the object which calculates the exact intersection
     * point with the next boundary.
     */
    inline G4VIntersectionLocator* GetIntersectionLocator();
    inline void SetIntersectionLocator(G4VIntersectionLocator* pLocator );

    /**
     * Controls the parameter which enables the temporary 'relaxation' which
     * ensures that chord segments are short enough so that their sagitta is
     * small than delta-chord parameter.
     * The Set method increases the value of delta-chord temporarily, doubling
     * it once the number of iterations substeps reach value of
     * 'IncreaseChordDistanceThreshold'. It is also doubled again every time
     * the iteration count reaches a multiple of this value.
     *   @note The delta-chord is reset to its original value at the end of
     *         each call to ComputeStep().
     */
    inline G4int GetIterationsToIncreaseChordDistance() const;
    inline void  SetIterationsToIncreaseChordDistance(G4int numIters);

    /**
     * Accessors.
     */
    inline G4double GetDeltaIntersection() const;
    inline G4double GetDeltaOneStep() const;

    /**
     * Auxiliary methods.
     *   @note Their results can/will change during propagation.
     */
    inline G4FieldManager* GetCurrentFieldManager();
    inline G4EquationOfMotion* GetCurrentEquationOfMotion();

    /**
     * Accessor and modifier for navigator.
     */
    inline void SetNavigatorForPropagating(G4Navigator* SimpleOrMultiNavigator); 
    inline G4Navigator* GetNavigatorForPropagating();

    /**
     * Accessors and modifiers for no-zero steps threshold.
     */
    inline void SetThresholdNoZeroStep( G4int noAct,
                                        G4int noHarsh,
                                        G4int noAbandon );
    inline G4int GetThresholdNoZeroSteps( G4int i ); 
    inline G4double GetZeroStepThreshold(); 
    inline void     SetZeroStepThreshold( G4double newLength ); 
   
    /**
     * Updates the Locator with parameters from this class and from current
     * field manager.
     */
    void RefreshIntersectionLocator(); 

  protected:

    /**
     * Logging methods.
     */
    void PrintStepLengthDiagnostic( G4double      currentProposedStepLength,
                                    G4double      decreaseFactor,
                                    G4double      stepTrial,
                              const G4FieldTrack& aFieldTrack);
    void ReportLoopingParticle( G4int count,  G4double StepTaken,
                                G4double stepRequest, const char* methodName,
                                const G4ThreeVector&      momentumVec,
                                G4VPhysicalVolume* physVol);
    void ReportStuckParticle(G4int noZeroSteps, G4double proposedStep,
                             G4double lastTriedStep, G4VPhysicalVolume* physVol);

  private:

    // ----------------------------------------------------------------------
    //  DATA Members
    // ----------------------------------------------------------------------

    //  ==================================================================
    //  INVARIANTS - Must not change during tracking

    //  ** PARAMETERS -----------
    G4int fMax_loop_count = 1000;
      // Limit for the number of sub-steps taken in one call to ComputeStep
    G4int fIncreaseChordDistanceThreshold = 100;
    G4bool fUseSafetyForOptimisation = true;
      // (false) is less sensitive to incorrect safety

    //  Thresholds for identifying "abnormal" cases - which cause looping
    //
    G4int fActionThreshold_NoZeroSteps = 2;        // Threshold # - above it act
    G4int fSevereActionThreshold_NoZeroSteps = 10; // Threshold # to act harshly
    G4int fAbandonThreshold_NoZeroSteps = 50;      // Threshold # to abandon
    G4double fZeroStepThreshold = 0.0; 
      // Threshold *length* for counting of tiny or 'zero' steps 

    // Parameters related to handling of very large steps which
    // occur typically in large volumes with vacuum or very thin gas
    //
    G4double fLargestAcceptableStep;
      // Maximum size of a step - for optimization (and to avoid problems)
    G4double fMaxStepSizeMultiplier = 3;
      // Multiplier for directional exit distance used as extra long-step limit 
    G4double fMinBigDistance= 100. ; // * CLHEP::mm
      // Minimum distance added to directional exit distance
    //  ** End of PARAMETERS -----

    G4double kCarTolerance;
        // Geometrical tolerance defining surface thickness

    G4bool fAllocatedLocator;                    //  Book-keeping

    //  --------------------------------------------------------
    //  ** Dependent Objects - to which work is delegated 

    G4FieldManager* fDetectorFieldMgr; 
      // The  Field Manager of the whole Detector.  (default)

    G4VIntersectionLocator* fIntersectionLocator;
      // Refines candidate intersection

    G4VCurvedTrajectoryFilter* fpTrajectoryFilter = nullptr;
      // The filter encapsulates the algorithm which selects which
      // intermediate points should be stored in a trajectory. 
      // When it is NULL, no intermediate points will be stored.
      // Else PIF::ComputeStep must submit (all) intermediate
      // points it calculates, to this filter.  (jacek 04/11/2002)

    G4Navigator* fNavigator;
      // Set externally - only by tracking / run manager
    //
    //  ** End of Dependent Objects ----------------------------

    //  End of INVARIANTS 
    //  ==================================================================

    //  STATE information
    //  -----------------
    G4FieldManager* fCurrentFieldMgr;
      // The  Field Manager of the current volume (may be the global)
    G4bool fSetFieldMgr = false;  // Has it been set for the current step?

    // Parameters of current step
    //
    G4double fEpsilonStep;            // Relative accuracy of current Step
    G4FieldTrack End_PointAndTangent; // End point storage
    G4bool fParticleIsLooping = false;
    G4int fNoZeroStep = 0;            // Count of zero Steps

    // State used for Optimisation
    //
    G4double fFull_CurveLen_of_LastAttempt = -1; 
    G4double fLast_ProposedStepLength = -1; 
      // Previous step information -- for use in adjust step size
    G4ThreeVector fPreviousSftOrigin;
    G4double fPreviousSafety = 0.0; 
      // Last safety origin & value: for optimisation

    G4int fVerboseLevel = 0;
    G4bool fVerbTracePiF = false;
    G4bool fCheck = false;
      // For debugging purposes

    G4bool fFirstStepInVolume = true; 
    G4bool fLastStepInVolume = true; 
    G4bool fNewTrack = true;
};

// Inline methods
//
#include "G4PropagatorInField.icc"

#endif 
