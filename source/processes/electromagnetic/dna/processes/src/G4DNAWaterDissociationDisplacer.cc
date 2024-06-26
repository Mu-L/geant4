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
//
// Author: Mathieu Karamitros
//
// WARNING : This class is released as a prototype.
// It might strongly evolve or even disapear in the next releases.
//
// History:
// -----------
// 10 Oct 2011 M.Karamitros created
//
// -------------------------------------------------------------------

#include "G4DNAWaterDissociationDisplacer.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4H2O.hh"
#include "G4H2.hh"
#include "G4Hydrogen.hh"
#include "G4Oxygen.hh"
#include "G4OH.hh"
#include "G4H3O.hh"
#include "G4Electron_aq.hh"
#include "G4H2O2.hh"
#include "Randomize.hh"
#include "G4Molecule.hh"
#include "G4MolecularConfiguration.hh"
#include "G4RandomDirection.hh"
#include "G4Exp.hh"
#include "G4UnitsTable.hh"
#include "G4EmParameters.hh"
#include "G4DNAOneStepThermalizationModel.hh"

using namespace std;

//------------------------------------------------------------------------------

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                Ionisation_DissociationDecay)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                A1B1_DissociationDecay)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                B1A1_DissociationDecay)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                B1A1_DissociationDecay2)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                AutoIonisation)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                DissociativeAttachment)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                DoubleIonisation_DissociationDecay1)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                DoubleIonisation_DissociationDecay2)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                DoubleIonisation_DissociationDecay3)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                TripleIonisation_DissociationDecay)

G4CT_COUNT_IMPL(G4DNAWaterDissociationDisplacer,
                QuadrupleIonisation_DissociationDecay)

/*
//------------------------------------------------------------------------------
#ifdef _WATER_DISPLACER_USE_KREIPL_
// This function is used to generate the following density distribution:
//   f(r) := 4*r * exp(-2*r)
// reference:
//  Kreipl, M. S., Friedland, W. & Paretzke, H. G.
//  Time- and space-resolved Monte Carlo study of water radiolysis for photon,
//  electron and ion irradiation.
//  Radiat. Environ. Biophys. 48, 11–20 (2009).
G4double G4DNAWaterDissociationDisplacer::ElectronProbaDistribution(G4double r)
{
  return (2.*r+1.)*G4Exp(-2.*r);
}
#endif

//------------------------------------------------------------------------------
#ifdef _WATER_DISPLACER_USE_TERRISOL_
// This function is used to generate the following density distribution:
//   f(r) := 4*x^2/(sqrt(%pi)*(b)^3)*exp(-x^2/(b)^2)
// with b=27.22 for 7 eV
// reference
// Terrissol M, Beaudre A (1990) Simulation of space and time evolution
// of radiolytic species induced by electrons in water.
// Radiat Prot Dosimetry 31:171–175

G4double G4DNAWaterDissociationDisplacer::ElectronProbaDistribution(G4double r)
{
#define b 27.22 // nanometer
    static constexpr double sqrt_pi = 1.77245; // sqrt(CLHEP::pi);
    static constexpr double b_to3 = 20168.1; // pow(b,3.);
    static constexpr double b_to2 = 740.928; // pow(b,2.);
    static constexpr double inverse_b_to2 = 1. / b_to2;

    static constexpr double main_factor = 4. / (sqrt_pi * b_to3);
    static constexpr double factorA = sqrt_pi * b_to3 / 4.;
    static constexpr double factorB = b_to2 / 2.;

    return (main_factor *
            (factorA * erf(r / b)
             - factorB * r * G4Exp(-pow(r, 2.) * inverse_b_to2)));
}

#endif
//------------------------------------------------------------------------------
*/
G4DNAWaterDissociationDisplacer::G4DNAWaterDissociationDisplacer()
        :

        ke(1.7*eV)
/*#ifdef _WATER_DISPLACER_USE_KREIPL_
        fFastElectronDistrib(0., 5., 0.001)
#elif defined _WATER_DISPLACER_USE_TERRISOL_
        fFastElectronDistrib(0., 100., 0.001)
#endif*/
{/*
    fProba1DFunction =
            std::bind((G4double(*)(G4double))
                              &G4DNAWaterDissociationDisplacer::ElectronProbaDistribution,
                      std::placeholders::_1);

    size_t nBins = 500;
    fElectronThermalization.reserve(nBins);
    double eps = 1. / ((int) nBins);
    double proba = eps;

    fElectronThermalization.push_back(0.);

    for (size_t i = 1; i < nBins; ++i)
    {
        double r = fFastElectronDistrib.Revert(proba, fProba1DFunction);
        fElectronThermalization.push_back(r * nanometer);
        proba += eps;
//  G4cout << G4BestUnit(r*nanometer, "Length") << G4endl;
    }*/
  dnaSubType = G4EmParameters::Instance()->DNAeSolvationSubType();
//   SetVerbose(1);
}

//------------------------------------------------------------------------------

G4DNAWaterDissociationDisplacer::~G4DNAWaterDissociationDisplacer()
{
}

//------------------------------------------------------------------------------

G4ThreeVector G4DNAWaterDissociationDisplacer::GetMotherMoleculeDisplacement(
  const G4MolecularDissociationChannel* theDecayChannel) const
{
  G4int decayType = theDecayChannel->GetDisplacementType();
  G4double RMSMotherMoleculeDisplacement(0.0);

  switch (decayType) {
    case Ionisation_DissociationDecay:
      RMSMotherMoleculeDisplacement = 2.0 * nanometer;
      break;
    case A1B1_DissociationDecay:
      RMSMotherMoleculeDisplacement = 0.0 * nanometer;
      break;
    case B1A1_DissociationDecay:
      RMSMotherMoleculeDisplacement = 0.0 * nanometer;
      break;
    case B1A1_DissociationDecay2:
      RMSMotherMoleculeDisplacement = 0.0 * nanometer;
      break;
    case AutoIonisation:
      RMSMotherMoleculeDisplacement = 2.0 * nanometer;
      break;
    case DissociativeAttachment:
      RMSMotherMoleculeDisplacement = 0.0 * nanometer;
      break;
    case DoubleIonisation_DissociationDecay1:
      RMSMotherMoleculeDisplacement = 2.0 * nanometer;
      break;
    case DoubleIonisation_DissociationDecay2:
      RMSMotherMoleculeDisplacement = 2.0 * nanometer;
      break;
    case DoubleIonisation_DissociationDecay3:
      RMSMotherMoleculeDisplacement = 2.0 * nanometer;
      break;
    case TripleIonisation_DissociationDecay:
      RMSMotherMoleculeDisplacement = 2.0 * nanometer;
      break;
    case QuadrupleIonisation_DissociationDecay:
      RMSMotherMoleculeDisplacement = 2.0 * nanometer;
      break;
  }

  if (RMSMotherMoleculeDisplacement == 0) {
    return G4ThreeVector(0, 0, 0);
  }

  auto RandDirection =
    radialDistributionOfProducts(RMSMotherMoleculeDisplacement);

  return RandDirection;
}

//------------------------------------------------------------------------------

vector<G4ThreeVector> G4DNAWaterDissociationDisplacer::GetProductsDisplacement(
  const G4MolecularDissociationChannel* pDecayChannel) const
{
  G4int nbProducts = pDecayChannel->GetNbProducts();
  vector<G4ThreeVector> theProductDisplacementVector(nbProducts);

  typedef map<const G4MoleculeDefinition*, G4double> RMSmap;
  RMSmap theRMSmap;

  G4int decayType = pDecayChannel->GetDisplacementType();

  switch (decayType) {
    case Ionisation_DissociationDecay:
      {
        if (fVerbose != 0) {
          G4cout << "Ionisation_DissociationDecay" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }
        G4double RdmValue = G4UniformRand();

        if (RdmValue < 0.5) {
          // H3O
          theRMSmap[G4H3O::Definition()] = 0.0 * nanometer;
          // OH
          theRMSmap[G4OH::Definition()]  = 0.8 * nanometer;
        } else {
          // H3O
          theRMSmap[G4H3O::Definition()] = 0.8 * nanometer;
          // OH
          theRMSmap[G4OH::Definition()]  = 0.0 * nanometer;
        }

        for (G4int i = 0; i < nbProducts; i++) {
          auto pProduct = pDecayChannel->GetProduct(i);
          G4double theRMSDisplacement = theRMSmap[pProduct->GetDefinition()];

          if (theRMSDisplacement == 0.0) {
            theProductDisplacementVector[i] = G4ThreeVector();
          } else {
            auto RandDirection =
              radialDistributionOfProducts(theRMSDisplacement);
            theProductDisplacementVector[i] = RandDirection;
          }
        }
        break;
      }
    case A1B1_DissociationDecay:
      {
        if (fVerbose != 0) {
          G4cout << "A1B1_DissociationDecay" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }

        constexpr G4double theRMSDisplacement = 2.4 * nanometer;
        auto RandDirection = radialDistributionOfProducts(theRMSDisplacement);

        for (G4int i = 0; i < nbProducts; i++) {
          auto pProduct = pDecayChannel->GetProduct(i);

          if (pProduct->GetDefinition() == G4OH::Definition()) {
            theProductDisplacementVector[i] =  -1.0 / 18.0 * RandDirection;
          } else if (pProduct->GetDefinition() == G4Hydrogen::Definition()) {
            theProductDisplacementVector[i] = +17.0 / 18.0 * RandDirection;
          }
        }
        break;
      }
    case B1A1_DissociationDecay:
      {
        if (fVerbose != 0) {
          G4cout << "B1A1_DissociationDecay" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }

        constexpr G4double theRMSDisplacement = 0.8 * nanometer;
        auto RandDirection = radialDistributionOfProducts(theRMSDisplacement);

        G4int NbOfOH = 0;
        for (G4int i = 0; i < nbProducts; ++i) {
          auto pProduct = pDecayChannel->GetProduct(i);
          if (pProduct->GetDefinition() == G4H2::Definition()) {
            // In the paper of Kreipl (2009)
            // theProductDisplacementVector[i] = -2.0 / 18.0 * RandDirection;

            // Based on momentum conservation
            theProductDisplacementVector[i] = -16.0 / 18.0 * RandDirection;
          } else if (pProduct->GetDefinition() == G4OH::Definition()) {
            // In the paper of Kreipl (2009)
            // G4ThreeVector OxygenDisplacement = +16.0 / 18.0 * RandDirection;

            // Based on momentum conservation
            G4ThreeVector OxygenDisplacement = +2.0 / 18.0 * RandDirection;
            constexpr G4double OHRMSDisplacement = 1.1 * nanometer;

            auto OHDisplacement =
              radialDistributionOfProducts(OHRMSDisplacement);

            if (NbOfOH == 0) {
              OHDisplacement =  0.5 * OHDisplacement;
            } else {
              OHDisplacement = -0.5 * OHDisplacement;
            }

            theProductDisplacementVector[i] =
              OHDisplacement + OxygenDisplacement;

            ++NbOfOH;
          }
        }
        break;
      }
    case B1A1_DissociationDecay2:
      {
        if (fVerbose != 0){
          G4cout << "B1A1_DissociationDecay2" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }

        G4int NbOfH = 0;
        for (G4int i = 0; i < nbProducts; ++i) {
          auto pProduct = pDecayChannel->GetProduct(i);
          if (pProduct->GetDefinition() == G4Oxygen::Definition()) {
            // O(3p)
            theProductDisplacementVector[i] = G4ThreeVector(0, 0, 0);
          } else if (pProduct->GetDefinition() == G4Hydrogen::Definition()) {
            // H
            constexpr G4double HRMSDisplacement = 1.6 * nanometer;

            auto HDisplacement =
              radialDistributionOfProducts(HRMSDisplacement);

            if (NbOfH == 0) {
              HDisplacement =  0.5 * HDisplacement;
            } else {
              HDisplacement = -0.5 * HDisplacement;
            }
            theProductDisplacementVector[i] = HDisplacement;

            ++NbOfH;
          }
        }
        break;
      }
    case AutoIonisation:
      {
        if (fVerbose != 0) {
          G4cout << "AutoIonisation" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }

        G4double RdmValue = G4UniformRand();

        if (RdmValue < 0.5) {
          // H3O
          theRMSmap[G4H3O::Definition()] = 0.0 * nanometer;
          // OH
          theRMSmap[G4OH::Definition()]  = 0.8 * nanometer;
        } else {
          // H3O
          theRMSmap[G4H3O::Definition()] = 0.8 * nanometer;
          // OH
          theRMSmap[G4OH::Definition()]  = 0.0 * nanometer;
        }

        for (G4int i = 0; i < nbProducts; i++) {
          auto pProduct = pDecayChannel->GetProduct(i);
          auto theRMSDisplacement = theRMSmap[pProduct->GetDefinition()];

          if (theRMSDisplacement == 0) {
            theProductDisplacementVector[i] = G4ThreeVector();
          } else {
            auto RandDirection =
              radialDistributionOfProducts(theRMSDisplacement);
            theProductDisplacementVector[i] = RandDirection;
          }
          if (pProduct->GetDefinition() == G4Electron_aq::Definition()) {
            theProductDisplacementVector[i] = radialDistributionOfElectron();
          }
        }
        break;
      }
    case DissociativeAttachment:
      {
        if (fVerbose != 0) {
          G4cout << "DissociativeAttachment" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }
        constexpr G4double theRMSDisplacement = 0.8 * nanometer;
        auto RandDirection = radialDistributionOfProducts(theRMSDisplacement);

        G4int NbOfOH = 0;
        for (G4int i = 0; i < nbProducts; ++i) {
          auto pProduct = pDecayChannel->GetProduct(i);
          if (pProduct->GetDefinition() == G4H2::Definition()) {
            // In the paper of Kreipl (2009)
            // theProductDisplacementVector[i] = -2.0 / 18.0 * RandDirection;

            // Based on momentum conservation
            theProductDisplacementVector[i] = -16.0 / 18.0 * RandDirection;
          } else if (pProduct->GetDefinition() == G4OH::Definition()) {
            // In the paper of Kreipl (2009)
            // G4ThreeVector OxygenDisplacement = +16.0 / 18.0 * RandDirection;

            // Based on momentum conservation
            G4ThreeVector OxygenDisplacement = +2.0 / 18.0 * RandDirection;
            constexpr G4double OHRMSDisplacement = 1.1 * nanometer;

            auto OHDisplacement =
              radialDistributionOfProducts(OHRMSDisplacement);

            if (NbOfOH == 0) {
              OHDisplacement =  0.5 * OHDisplacement;
            } else {
              OHDisplacement = -0.5 * OHDisplacement;
            }
            theProductDisplacementVector[i] = OHDisplacement +
              OxygenDisplacement;
            ++NbOfOH;
          }
        }
        break;
      }
    case DoubleIonisation_DissociationDecay1:
      {
        if (fVerbose != 0) {
          G4cout << "DoubleIonisation_DissociationDecay1" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }

        // Ref.) B. Gervais, et al., DOI: 10.1016/j.radphyschem.2005.09.01
        // Decay Channel #1: H2O^2+ -> 2H+ + O(3P) -> 2H3O+ + O(3P)

        theRMSmap[G4H3O::Definition()]    = 1.2 * nanometer;
        theRMSmap[G4Oxygen::Definition()] = 0.0 * nanometer;

        for (G4int i = 0, num_H3O = 0; i < nbProducts; i++) {

          const auto prod_def = pDecayChannel->GetProduct(i)->GetDefinition();

          if (prod_def == G4H3O::Definition()) {
            num_H3O++;
            if (num_H3O == 2) {
              constexpr G4double H3Op_rms = 0.3 * nanometer;
              theRMSmap[G4H3O::Definition()] = H3Op_rms;
            }
          }

          theProductDisplacementVector[i]
            = radialDistributionOfProducts(theRMSmap[prod_def]);

        }

        break;
      }
    case DoubleIonisation_DissociationDecay2:
      {
        if (fVerbose != 0) {
          G4cout << "DoubleIonisation_DissociationDecay2" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }

        // Ref.) B. Gervais, et al., DOI: 10.1016/j.radphyschem.2005.09.01
        // Decay Channel #2: H2O^2+ -> H+ + H* + O+ -> 2H3O+ + H* + *OH + O(3P)

        theRMSmap[G4H3O::Definition()]      = 1.2 * nanometer;
        theRMSmap[G4Oxygen::Definition()]   = 0.0 * nanometer;
        theRMSmap[G4Hydrogen::Definition()] = 0.8 * nanometer;
        theRMSmap[G4OH::Definition()]       = 0.3 * nanometer;

        const auto OH_disp = radialDistributionOfProducts(
                                theRMSmap[G4OH::Definition()]);

        for (G4int i = 0, num_H3O = 0; i < nbProducts; i++) {

          const auto prod_def = pDecayChannel->GetProduct(i)->GetDefinition();

          if (prod_def == G4H3O::Definition()) {
            num_H3O++;
            if (num_H3O == 2) {
              constexpr G4double OH_rms = 0.3 * nanometer;
              theProductDisplacementVector[i]
                = radialDistributionOfProducts(OH_rms);
              theProductDisplacementVector[i] += OH_disp;
              continue;
            }
          } else if (prod_def == G4OH::Definition()) {
            theProductDisplacementVector[i] = OH_disp;
            continue;
          }

          theProductDisplacementVector[i]
            = radialDistributionOfProducts(theRMSmap[prod_def]);

        }

        break;
      }
    case DoubleIonisation_DissociationDecay3:
      {
        if (fVerbose != 0) {
          G4cout << "DoubleIonisation_DissociationDecay3" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }

        // Ref.) B. Gervais, et al., DOI: 10.1016/j.radphyschem.2005.09.01
        // Decay Channel #3: H2O^2+ -> H+ + OH+ -> 2H3O+ + O(3P)

        theRMSmap[G4H3O::Definition()]    = 1.2 * nanometer;
        theRMSmap[G4Oxygen::Definition()] = 0.0 * nanometer;

        for (G4int i = 0; i < nbProducts; i++) {

          const auto prod_def = pDecayChannel->GetProduct(i)->GetDefinition();

          theProductDisplacementVector[i]
            = radialDistributionOfProducts(theRMSmap[prod_def]);

        }

        break;
      }
    case TripleIonisation_DissociationDecay:
      {
        if (fVerbose != 0) {
          G4cout << "TripleIonisation_DissociationDecay" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }

        // Ref.) B. Gervais, et al., DOI: 10.1016/j.radphyschem.2005.09.01
        // Decay Channel: H2O^3+ -> 3H3O+ + *OH + O(3P)
        //
        // ** Detaied decay chain **
        //    H2O^3+ -> O+ + H3O+ + H3O+
        //              |
        //              --> H2O+ + O(3P)
        //                  |
        //                  --> *OH + H3O+

        theRMSmap[G4H3O::Definition()]    = 1.2 * nanometer;
        theRMSmap[G4Oxygen::Definition()] = 0.0 * nanometer;
        theRMSmap[G4OH::Definition()]     = 0.3 * nanometer;

        const auto OH_disp = radialDistributionOfProducts(
            theRMSmap[G4OH::Definition()]);

        for (G4int i = 0, num_H3O = 0; i < nbProducts; i++) {

          const auto prod_def = pDecayChannel->GetProduct(i)->GetDefinition();

          if (prod_def == G4H3O::Definition()) {
            num_H3O++;
            if (num_H3O == 3) {
              constexpr G4double H3Op_rms = 0.3 * nanometer;
              theProductDisplacementVector[i]
                = radialDistributionOfProducts(H3Op_rms);
              theProductDisplacementVector[i] += OH_disp;
              continue;
            }
          } else if (prod_def == G4OH::Definition()) {
            theProductDisplacementVector[i] = OH_disp;
            continue;
          }

          theProductDisplacementVector[i]
            = radialDistributionOfProducts(theRMSmap[prod_def]);

        }

        break;
      }
    case QuadrupleIonisation_DissociationDecay:
      {
        if (fVerbose != 0) {
          G4cout << "QuadrupleIonisation_DissociationDecay" << G4endl;
          G4cout << "Channel's name: " << pDecayChannel->GetName() << G4endl;
        }

        // Ref.) B. Gervais, et al., DOI: 10.1016/j.radphyschem.2005.09.01
        // Decay Channel: H2O^4+ -> 4H3O+ + 2*OH + O(3P)
        //
        // ** Detaied decay chain **
        //    H2O^4+ -> O^2+ + H3O+ + H3O+
        //              |
        //              --> H2O+ + H2O+ + O(3P)
        //                  |      |
        //                  |      --> *OH + H3O+
        //                  |
        //                  --> *OH + H3O+

        theRMSmap[G4H3O::Definition()]    = 1.2 * nanometer;
        theRMSmap[G4Oxygen::Definition()] = 0.0 * nanometer;
        theRMSmap[G4OH::Definition()]     = 0.3 * nanometer;

        const auto OH_disp1 = radialDistributionOfProducts(
            theRMSmap[G4OH::Definition()]);
        const auto OH_disp2 = radialDistributionOfProducts(
            theRMSmap[G4OH::Definition()]);

        for (G4int i = 0, num_H3O = 0, num_OH = 0; i < nbProducts; i++) {

          const auto prod_def = pDecayChannel->GetProduct(i)->GetDefinition();

          if (prod_def == G4H3O::Definition()) {

            num_H3O++;
            constexpr G4double H3Op_rms = 0.3 * nanometer;

            if (num_H3O == 3) {
              theProductDisplacementVector[i]
                = radialDistributionOfProducts(H3Op_rms);
              theProductDisplacementVector[i] += OH_disp1;
              continue;
            } else if (num_H3O == 4) {
              theProductDisplacementVector[i]
                = radialDistributionOfProducts(H3Op_rms);
              theProductDisplacementVector[i] += OH_disp2;
              continue;
            }

          } else if (prod_def == G4OH::Definition()) {

            num_OH++;
            if (num_OH == 1) {
              theProductDisplacementVector[i] = OH_disp1;
            } else {
              theProductDisplacementVector[i] = OH_disp2;
            }

            continue;
          }

          theProductDisplacementVector[i]
            = radialDistributionOfProducts(theRMSmap[prod_def]);

        }

        break;
      }

  }
  return theProductDisplacementVector;
}

//------------------------------------------------------------------------------

G4ThreeVector
G4DNAWaterDissociationDisplacer::
radialDistributionOfProducts(G4double Rrms) const
{
  static const double inverse_sqrt_3 = 1.0 / sqrt(3.0);
  double sigma = Rrms * inverse_sqrt_3;
  double x = G4RandGauss::shoot(0.0, sigma);
  double y = G4RandGauss::shoot(0.0, sigma);
  double z = G4RandGauss::shoot(0.0, sigma);
  return G4ThreeVector(x, y, z);
}

//------------------------------------------------------------------------------

G4ThreeVector
G4DNAWaterDissociationDisplacer::radialDistributionOfElectron() const
{/*
    G4double rand_value = G4UniformRand();
    size_t nBins = fElectronThermalization.size();
    size_t bin = size_t(floor(rand_value * nBins));
    size_t bin_p1 = min(bin + 1, nBins - 1);

    return (fElectronThermalization[bin] * (1. - rand_value)
            + fElectronThermalization[bin_p1] * rand_value) *
           G4RandomDirection();*/

  G4ThreeVector pdf = G4ThreeVector(0, 0, 0);

  if (dnaSubType == fRitchie1994eSolvation) {
    DNA::Penetration::Ritchie1994::GetPenetration(ke, pdf);
  } else if (dnaSubType == fTerrisol1990eSolvation) {
    DNA::Penetration::Terrisol1990::GetPenetration(ke, pdf);
  } else if (dnaSubType == fMeesungnoensolid2002eSolvation) {
    DNA::Penetration::Meesungnoen2002_amorphous::GetPenetration(ke, pdf);
  } else if (dnaSubType == fKreipl2009eSolvation) {
    DNA::Penetration::Kreipl2009::GetPenetration(ke, pdf);
  } else {
    DNA::Penetration::Meesungnoen2002::GetPenetration(ke, pdf);
  }
  return pdf;
}
